import torch
from torch import nn
import cv2
from transformers import BertTokenizer
from torch.nn import functional as F
from collections import OrderedDict
import time
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


torch.set_float32_matmul_precision('high')

def create_loss_fn(loss_fn_type):
    if loss_fn_type == "mse":
        return nn.MSELoss()
    elif loss_fn_type == "mae":
        return nn.L1Loss()
    elif "huber" in loss_fn_type:
        return nn.SmoothL1Loss(float(loss_fn_type.split(":")[-1]))
    elif loss_fn_type == "pairwise":
        return nn.MarginRankingLoss()
    elif loss_fn_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid loss function type")

class AttentionBlock(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, num_heads=8):
        super(AttentionBlock, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value):
        cross_attention_output, cross_attention_weights = self.cross_attention(query, key, value)
        cross_attention_output = self.norm1(cross_attention_output + query)
        ffn_output = self.ffn(cross_attention_output)
        ffn_output = self.norm2(ffn_output + cross_attention_output)
        
        return ffn_output, cross_attention_weights
        

class FeatureFusionBlock(nn.Module):
    def __init__(self, d_model=256, num_heads=8, fusion_blocks_type="cross-attn", no_action_input=False):
        super(FeatureFusionBlock, self).__init__()
        self.d_model = d_model
        self.no_action_input = no_action_input
        self.fusion_blocks_type = fusion_blocks_type
        if self.fusion_blocks_type == "cross-attn":
            self.image_cross_attention = AttentionBlock(d_model=d_model, num_heads=num_heads)
            self.text_cross_attention = AttentionBlock(d_model=d_model, num_heads=num_heads)
            self.self_attention = AttentionBlock(d_model=d_model, num_heads=num_heads)
        elif self.fusion_blocks_type == "self-attn":
            self.self_attention = AttentionBlock(d_model=d_model, num_heads=num_heads)
    
    def forward(self, image_feature, text_feature, action_query):
        if self.fusion_blocks_type == "cross-attn":
            action_query, _ = self.image_cross_attention(action_query, image_feature, image_feature)
            action_query, _ = self.text_cross_attention(action_query, text_feature, text_feature) if text_feature is not None else (action_query, None)
            action_query, _ = self.self_attention(action_query, action_query, action_query)
            return image_feature, text_feature, action_query
        elif self.fusion_blocks_type == "self-attn":
            if text_feature is not None:
                concated_feature = torch.cat([image_feature, text_feature, action_query], dim=1)
                concated_feature, _ = self.self_attention(concated_feature, concated_feature, concated_feature)
                image_feature, text_feature, action_query = torch.split(concated_feature, [image_feature.shape[1], text_feature.shape[1], action_query.shape[1]], dim=1)
            else:
                concated_feature = torch.cat([image_feature, action_query], dim=1)
                concated_feature, _ = self.self_attention(concated_feature, concated_feature, concated_feature)
                image_feature, action_query = torch.split(concated_feature, [image_feature.shape[1], action_query.shape[1]], dim=1)
            return image_feature, text_feature, action_query
             
        
class RegressionHead(nn.Module):
    def __init__(self, d_model=256, output_dim=1, head_token="cls"):
        super(RegressionHead, self).__init__()
        self.linear = nn.Linear(d_model, output_dim)
        self.head_token = head_token
        
    def forward(self, x):
        if self.head_token == "cls":
            token = x[:, 0, :]
        elif self.head_token == "mean":
            token = torch.mean(x, dim=1)
            
        output = self.linear(token)
        
        return output

class RankHead(nn.Module):
    def __init__(self, d_model=256, output_dim=1, head_token="cls"):
        super(RankHead, self).__init__()
        self.head_token = head_token
        self.linear = nn.Linear(d_model, output_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        if self.head_token == "cls":
            token = x[:, 0, :]
        elif self.head_token == "mean":
            token = torch.mean(x, dim=1)
        output = self.linear(token)
        prob = self.softmax(output)
        return prob
    
def get_positional_encoding(n_position, d_model):
    pe = torch.zeros(n_position, d_model)
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
        
class Discriminator(nn.Module):
    def __init__(self, 
                 model_cfg_path, 
                 weights_path, 
                 action_query_length=64, 
                 no_action_input=False,
                 no_text_input=False,
                 frozen_encoder=True,
                 num_blocks=10, 
                 fusion_blocks_type="cross-attn",
                 encoder_type="groundingdino",
                 window_size=1,
                 action_horizon=1,
                 future_action=0,
                 future_image=False,
                 histroy_dropout_rate=0.1,
                 use_norm_delta_proprio=False,
                 use_delta_grip_act=False,
                 d_model=256,
                 n_heads=8,
                 head_type="regression",
                 head_token="cls",
                 num_ranks=None,
                 rank_thres=None,
                 loss_fn_type="mse",
                 normalize_linear_combination=False,
                 activate_action_embedding=True,
                 metrics={"mse": True, "mae": True, "rank": [2, 3]},
                 eval_metrics={"mse": False, "mae": False, "rank": [3]},
                 cat_start_feature=False,
                 ):
        
        super(Discriminator, self).__init__()
        self.window_size = window_size
        self.action_horizon = action_horizon
        self.future_action = future_action
        self.future_image = future_image
        self.num_images = window_size + int(future_image)
        self.history_dropout_rate = histroy_dropout_rate
        self.d_model = d_model
        self.no_action_input = no_action_input
        self.no_text_input = no_text_input
        self.cat_start_feature = cat_start_feature
        if encoder_type == "groundingdino":
            from curation.suboptimal_classifier.discriminator.dino_feature_extractor import load_extractor_model, DinoFeatureExtractor
            self.dino_feature_extractor:DinoFeatureExtractor = load_extractor_model(model_cfg_path, weights_path)
            self.encoder = self.dino_feature_extractor
            self.feature_fusion_blocks = nn.ModuleList([FeatureFusionBlock(d_model=d_model, num_heads=n_heads, fusion_blocks_type=fusion_blocks_type) for _ in range(num_blocks)])
            
        elif encoder_type == "resnet":
            from torchvision.models import resnet101
            self.encoder = resnet101(pretrained=True)
            self.encoder.fc = nn.Identity()
            self.feature_fusion_blocks= nn.Sequential(
                nn.Linear(2048*self.num_images+action_query_length*d_model, d_model), # resnet101
                nn.LayerNorm(d_model),
                nn.ReLU(),
            )
        elif encoder_type == "dinov2":
            self.encoder = AutoModel.from_pretrained('facebook/dinov2-base')
            self.feature_fusion_blocks = nn.ModuleList([FeatureFusionBlock(d_model=d_model, num_heads=n_heads, fusion_blocks_type=fusion_blocks_type) for _ in range(num_blocks)])
        elif encoder_type == "radio":
            self.encoder =  AutoModel.from_pretrained('nvidia/RADIO-B', trust_remote_code=True)
            self.feature_fusion_blocks = nn.ModuleList([FeatureFusionBlock(d_model=d_model, num_heads=n_heads, fusion_blocks_type=fusion_blocks_type) for _ in range(num_blocks)])
            
        if encoder_type != "groundingdino" and not no_text_input:
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
            self.text_encoder = AutoModel.from_pretrained("distilbert/distilbert-base-uncased", torch_dtype=torch.float16)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            
        if frozen_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        self.encoder_type = encoder_type
        
        self.action_dim = 7
        self.action_dim += int(use_norm_delta_proprio) + int(use_delta_grip_act)
        self.use_norm_delta_proprio = use_norm_delta_proprio
        self.use_delta_grip_act = use_delta_grip_act
        self.action_embedding = nn.Parameter(torch.empty(self.action_dim, d_model, requires_grad=True))
        self.action_positional_encoding = nn.Parameter(get_positional_encoding(self.action_dim, d_model)*0.5, requires_grad=False)
        self.action_layer_norm = nn.LayerNorm(d_model)
        if action_horizon + future_action > 1:
            self.action_frame_level_pos_emb = nn.Parameter(get_positional_encoding(action_horizon + future_action, d_model)*0.5, requires_grad=False)
        nn.init.xavier_uniform_(self.action_embedding)
        
        self.normalize_linear_combination = normalize_linear_combination
        self.activate_action_embedding = activate_action_embedding
        self.action_embedding_bias = nn.Parameter(torch.zeros(action_query_length, d_model, requires_grad=True))
        self.linear_combination = nn.Parameter(torch.normal(size=(action_query_length, self.action_dim), dtype=torch.float32, mean=0, std=1, requires_grad=True))
        
        self.action_query_lenth = action_query_length
        self.fusion_blocks_type = fusion_blocks_type
        if self.num_images > 1:
            self.image_frame_level_pos_emb = nn.Parameter(get_positional_encoding(self.num_images, d_model)*0.5, requires_grad=True)
            self.image_layer_norm = nn.LayerNorm(d_model)
            self.text_layer_norm = nn.LayerNorm(d_model)
            if self.cat_start_feature:
                self.image_layer_norm2 = nn.LayerNorm(d_model)
                
            
        self.metric_pool = ThreadPoolExecutor()
        
        self.head_token = head_token
        if head_token == "cls":
            self.cls_token = nn.Parameter(torch.empty(size=(1, 1, d_model), dtype=torch.float32, requires_grad=True))
            nn.init.xavier_uniform_(self.cls_token)
        
        self.head_type = head_type
        if self.head_type == "rank":
            assert num_ranks is not None or rank_thres is not None, "num_ranks or rank_thres should be provided"
            # assert num_ranks is None or rank_thres is None, "num_ranks and rank_thres should not be provided at the same time"
            if num_ranks is not None:
                print("Overwrite rank_thres with num_ranks")
                self.rank_thres = torch.linspace(0, 1.0, num_ranks+1)[1:-1]
                self.num_ranks = num_ranks
            elif rank_thres is not None:
                self.rank_thres = torch.tensor(rank_thres)[1:, 0]
                self.num_ranks = len(self.rank_thres) + 1
                
        self.loss_fn_type = loss_fn_type.split(",")
        
        if head_type == "regression":
            self.head = RegressionHead(d_model=d_model, output_dim=1)
            for loss_fn_type in self.loss_fn_type:
                assert loss_fn_type in ["mse", "mae", "pairwise", "cross_entropy"] or "huber" in loss_fn_type
        else:
            self.head = RankHead(d_model=d_model, output_dim=self.num_ranks)
            for loss_fn_type in self.loss_fn_type:
                assert loss_fn_type == "cross_entropy"
        
        self.metrics = metrics
        self.eval_metrics = eval_metrics
               
    def get_image_text_features(self, image, text):
        if self.encoder_type == "resnet":
            image_feature = self.encoder(image)
            text_feature = None
        elif self.encoder_type == "groundingdino":
            image_feature, text_feature = self.encoder.get_enhanced_feaures(image, captions=text)
        elif self.encoder_type == "dinov2":
            image_feature = self.encoder(image)['last_hidden_state']
        elif self.encoder_type == "radio":
            _, image_feature = self.encoder(image)
        
        if self.encoder_type != "groundingdino" and self.no_text_input == False and text is not None:
            with torch.no_grad():
                if isinstance(text, list) or isinstance(text, tuple):
                    encoded_text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.encoder.device)
                    text_feature = self.text_encoder(**encoded_text)['last_hidden_state'] 
                elif isinstance(text, torch.Tensor):
                    text_feature = text.to(self.encoder.device)
        else:
            text_feature = None
        return image_feature, text_feature
    
    def get_action_features(self, action):
        '''
        action shape: (batch_size, H, 7)
        '''
        B, horizon, _ = action.shape
        action_embedding = action[..., None] * self.action_embedding[None, None, ...] # (batch_size, H, 7, 256)
        
        action_embedding = action_embedding + self.action_positional_encoding[None, None, ...] 
        
        action_embedding = self.action_layer_norm(action_embedding)
        
        if self.normalize_linear_combination:
            normalized_linear_combination = torch.nn.functional.normalize(self.linear_combination, p=2, dim=1)
        else:
            normalized_linear_combination = self.linear_combination
        
        action_query = normalized_linear_combination[None, None, ...] @ action_embedding + self.action_embedding_bias[None, ...] # (batch_size, H, T, 256)
        
        if self.activate_action_embedding:
            action_query = torch.nn.functional.gelu(action_query)
            
        if horizon > 1:
            action_query = action_query + self.action_frame_level_pos_emb[None, :, None, ...]
        # if in training, dropout history & future actions
        if self.training:
            drop_history = torch.rand((B, self.action_horizon-1), device=action_query.device) < self.history_dropout_rate
            action_query[:, :self.action_horizon-1, ...] = action_query[:, :self.action_horizon-1, ...] * ~drop_history[..., None, None]
            if self.future_action > 0:
                drop_future = torch.rand((B, self.future_action), device=action_query.device) < self.history_dropout_rate
                action_query[:, -self.future_action:, ...] = action_query[:, -self.future_action:, ...] * ~drop_future[..., None, None]
                
        action_query = action_query.view(B, horizon*self.action_query_lenth, self.d_model)
        
        
        if self.head_token == "cls":
            action_query = torch.cat([self.cls_token.repeat(action_query.shape[0], 1, 1), action_query], dim=1) # (batch_size, T+1, 256)
            
            
        if self.head_token == "cls":
            action_query = torch.cat([self.cls_token.repeat(action_query.shape[0], 1, 1), action_query], dim=1) # (batch_size, T+1, 256)
            
        return action_query
    
    def action_query_image_text_features(self, image, text, action):
        '''
        image shape: (batch_size*T, 3, H, W)
        text: list of strings with length batch_size*T
        action: (batch_size, 7)
        '''
        B = image.shape[0]
        if self.no_action_input:
            action_query = self.action_embedding_bias[None, ...].repeat(B, 1, 1)
        else:
            action_query = self.get_action_features(action)
        if self.head_token == "cls":
            action_query = torch.cat([self.cls_token.repeat(B, 1, 1), action_query], dim=1)
            
        if self.encoder_type == "resnet":
            B, T, C, H, W = image.shape
            image = image.view(B*T, C, H, W)
            image_feature, _ = self.get_image_text_features(image, text)
            image_feature = image_feature.view(B, T, *image_feature.shape[1:])
            image_feature = image_feature.view(B, -1)
            action_feature = action_query.view(B, -1)
            fused_action_feature = torch.cat([image_feature, action_feature], dim=1)
            action_query = self.feature_fusion_blocks(fused_action_feature).view(B, 1, self.d_model)
            return action_query, image_feature, None
        
        elif self.encoder_type == "groundingdino" or self.encoder_type == "dinov2" or self.encoder_type == "radio":
            text_features, image_features = [], []
            for t in range(self.num_images):
                image_feature, text_feature = self.get_image_text_features(image[:, t], text)
                image_features.append(image_feature)
                text_features.append(text_feature)
            image_features = torch.stack(image_features, dim=1)
            text_features = torch.stack(text_features, dim=1) if text_feature is not None else None
            
            if self.num_images > 1:
                delta_features = image_features[:, 1:] - image_features[:, :-1]
                delta_features = self.image_layer_norm(delta_features)
                
                if self.cat_start_feature:
                    normed_start_feature = self.image_layer_norm2(image_features[:, :1]) 
                    image_features = torch.cat([normed_start_feature, delta_features], dim=1)
                else:
                    # image_features = delta_features
                    image_features = image_features
                
                image_features = image_features + self.image_frame_level_pos_emb[None, :, None, ...]
                # image_features = self.image_layer_norm(image_features)
                
                if text_features is not None:
                    text_features = text_features + self.image_frame_level_pos_emb[None, :, None, ...]
                    text_features = self.text_layer_norm(text_features)
            
            image_features = image_features.view(image_feature.shape[0], self.num_images*image_features.shape[2], *image_features.shape[3:])
            text_features = text_features.reshape(text_feature.shape[0], self.num_images*text_features.shape[2], *text_features.shape[3:]) if text_feature is not None else None
            
            for block in self.feature_fusion_blocks:
                image_feature, text_feature, action_query = block(image_features, text_features, action_query)
                
            return action_query, image_features, text_features
    
    def compute_loss(self, result, score):
        loss = 0
        for loss_fn_type in self.loss_fn_type:
            loss_fn = create_loss_fn(loss_fn_type)
            if self.head_type == "regression":
                if loss_fn_type == "pairwise":
                    if score.shape[1] == 1: continue
                    else:
                        y = (score[:, 0] - score[:, 1]).sign()
                        loss += loss_fn(result[:, 0], result[:, 1], y)
                elif loss_fn_type == "cross_entropy":
                    if score.shape[1] == 1: continue
                    cls = torch.argmax(score, dim=1)
                    loss += loss_fn(result, cls)
                else:
                    loss += loss_fn(result, score)
                    
                    
            elif self.head_type == "rank":
                self.rank_thres = self.rank_thres.to(score.device)
                rank = torch.bucketize(score, self.rank_thres, right=False)
                rank = rank.reshape(-1)
                result = result.reshape(-1, result.shape[-1])
                loss += loss_fn(result, rank)
            
        return loss
    
    def compute_metrics(self, result, score, training=False):
        result = torch.clamp(result, 0.0, 1.0)
        metrics = OrderedDict()
        eps = 1e-6
        metrics_list = self.metrics if training else self.eval_metrics
        if self.head_type == "regression":
            if 'cross_entropy' in self.loss_fn_type:
                result = torch.nn.functional.sigmoid(result)
            if metrics_list.get("mse", False):
                metrics["mse"] = F.mse_loss(result, score).cpu().item()
            if metrics_list.get("mae", False):
                metrics["mae"] = F.l1_loss(result, score).cpu().item()
            if metrics_list.get("rank", False):
                for rank in metrics_list["rank"]:
                    assert isinstance(rank, int) and rank > 1, "Rank should be an integer greater than 1"
                    metrics[f"rank_{rank}"] = OrderedDict()
                    bins = torch.linspace(0-eps, 1.0, rank+1, dtype=torch.float32, device=result.device)
                    bined_score = torch.bucketize(score, bins, right=False)
                    bined_result = torch.bucketize(result, bins, right=False)
                    correct = bined_score == bined_result
                    metrics[f"rank_{rank}"]["total_acc"] = torch.sum(correct).cpu().item() / len(score)
                    for i in range(1, rank+1):
                        text = f"acc:score=={i}" if i == 0 else f"acc:{bins[i-1]:.2f}<score<={bins[i]:.2f}"
                        is_current_rank = bined_score == i
                        metrics[f"rank_{rank}"][text] = (torch.sum(correct & is_current_rank) / (torch.sum(is_current_rank) + 1e-6)).cpu().item()
            if metrics_list.get("thres", False):
                metrics["thres_acc"] = OrderedDict()
                for thres in metrics_list["thres"]:
                    pred_cls = result > thres
                    score_cls = score > thres
                    metrics["thres_acc"][f"{thres}_total"] = (torch.sum(pred_cls == score_cls) / torch.sum(torch.ones_like(score))).cpu().item()
                    if training:
                        metrics["thres_acc"][f"S>{thres}"] = (torch.sum(pred_cls & score_cls) / (torch.sum(score_cls) + 1e-6)).cpu().item()
                        metrics["thres_acc"][f"S<{thres}"] = (torch.sum(~pred_cls & ~score_cls) / (torch.sum(~score_cls) + 1e-6)).cpu().item()
            
            metrics['0 Score Acc'] = (torch.sum((result < 0.8) & (score==0.0)) / torch.sum(score==0.0)).cpu().item()
            metrics['1 Score Acc'] = (torch.sum((result > 0.8) & (score==1.0)) / torch.sum(score==1.0)).cpu().item()
            
            num_bins = metrics_list.get("pred_dist_bins", -1)
            
            if num_bins > 0:
                metrics["Pred_score_distribution"] = OrderedDict()
                if training:
                    metrics["0_score_distribution"] = OrderedDict()
                ground_turth_is_zero = score == 0
                bined_score = torch.bucketize(result, torch.linspace(0-eps, 1.0, num_bins+1, device=result.device), right=False)
                for i in range(1, num_bins+1):
                    text = f"score=={i}" if i == 0 else f"{(i-1)/num_bins:.2f}<score<={i/num_bins:.2f}"
                    is_current_rank = bined_score == i
                    metrics["Pred_score_distribution"][text] = (torch.sum(is_current_rank) / len(score)).cpu().item()
                    if training:
                        metrics["0_score_distribution"][text] = (torch.sum(is_current_rank & ground_turth_is_zero) / (torch.sum(ground_turth_is_zero) + 1e-6)).cpu().item()
        
        if self.head_type == "rank":
            ground_truth_cls = torch.bucketize(score, self.rank_thres.to(score.device), right=False)
            result = result.reshape(-1, result.shape[-1])
            ground_truth_cls = ground_truth_cls.reshape(-1)
            cross_entropy = F.cross_entropy(result, ground_truth_cls)
            metrics["cross_entropy"] = cross_entropy.cpu().item()
            
            cls_result = torch.argmax(result, dim=1)
            overall_acc = torch.sum(cls_result == ground_truth_cls).cpu().item() / len(ground_truth_cls)
            metrics["acc"] = {'overall': overall_acc}
            metrics["Pred_cls_distribution"] = OrderedDict()
            for rank in range(self.num_ranks):
                is_current_rank = ground_truth_cls == rank
                if training:
                    metrics["acc"][f"rank_{rank}"] = (torch.sum((cls_result == ground_truth_cls) & is_current_rank) / (torch.sum(is_current_rank) + 1e-6)).cpu().item()   
                metrics["Pred_cls_distribution"][f"rank_{rank}"] = (torch.sum(cls_result == rank) / len(ground_truth_cls)).cpu().item()
                    
        return metrics
    
    def forward(self, image, text, action, norm_delta_proprio=None, delta_grip_act=None, score=None, training=True):
        '''
        image shape: (batch_size, T, 3, H, W)
        text shape: list of strings with length batch_size
        action shape: (batch_size, H, 7) only the action of last observation is needed
        score shape: (batch_size, 1) only the last score is needed
        '''
        result = {}
        if len(image.shape) == 5:
            B, T, C, H, W = image.shape
            score = score.view(B, 1) if score is not None else None
            multiple_future = False
            score_shape = (B, 1)
            
        elif len(image.shape) == 6 and self.future_image:
            B, num_future, T, C, H, W = image.shape
            multiple_future = True
            if 'pairwise' in self.loss_fn_type:
                assert num_future == 2, "Only support 2 future images for pairwise loss"
            image = image.view(B*num_future, T, C, H, W)
            score = score.view(B, num_future) if score is not None else None
            score_shape = (B, num_future)
        score_shape = score_shape + (self.num_ranks,) if self.head_type == "rank" else score_shape
            
        assert T == self.num_images, f"Expected number of images {self.num_images} per frame, but got {T}"
        if not self.no_action_input:
            _, horizon, D = action.shape
            assert horizon == self.action_horizon + self.future_action, f"Expected action horizon {self.action_horizon + self.future_action}, but got {horizon}"
        if self.use_norm_delta_proprio:
            assert norm_delta_proprio is not None, "norm_delta_proprio should be provided because use_norm_delta_proprio is True"
            norm_delta_proprio = norm_delta_proprio.view(B, horizon, 1)
            action = torch.cat([action, norm_delta_proprio], dim=-1)
        
        if self.use_delta_grip_act:
            assert delta_grip_act is not None, "delta_grip_act should be provided because use_delta_grip_act is True"
            delta_grip_act = delta_grip_act.view(B, horizon, 1)
            action = torch.cat([action, delta_grip_act], dim=-1)
        
        if multiple_future:
            if not self.no_text_input:
                text += text
            if not self.no_action_input:
                action = action.repeat(num_future, 1, 1)
            
        fused_action_feature, image_feature, text_feature = self.action_query_image_text_features(image, text, action)
        result["output"] = self.head(fused_action_feature).view(*score_shape)
        if score is not None:
            loss = self.compute_loss(result["output"], score)
            metrics = self.compute_metrics(result["output"].detach().cpu(), score.detach().cpu(), training)
            result["loss"] = loss
            result["metrics"] = metrics
            
        return result

        
        
        

if __name__ == "__main__":
    device = torch.device("cuda")
    
    descriminator = Discriminator("GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", "GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")
    descriminator = descriminator.to(device)
    # breakpoint()
    IMAGE_PATH = "bridge_images/test.png"
    TEXT_PROMPT = "robot arm . gripper . toy . spoon . carrot . cloth . corn . small object"
    batch_size = 4
    _, image = load_image(IMAGE_PATH)
    action_query = descriminator.get_action_features(torch.rand((batch_size, 7)).to(device))
    rand_action = torch.rand((batch_size, 7)).to(device)
    batch_image = torch.stack([image for _ in range(batch_size)], dim=0).to(device)
    batched_prompts = [TEXT_PROMPT for _ in range(batch_size)]
    scores = torch.rand((batch_size, 1)).to(device)
    
    # descriminator.forward = torch.compile(descriminator.forward)
    
    with torch.no_grad():
        result = descriminator(batch_image, batched_prompts, rand_action, scores)
    import time
    from tqdm import tqdm
    start = time.time()
    timestep = 30
    for _ in tqdm(range(timestep)):
        result = descriminator(batch_image, batched_prompts, rand_action, scores)
        loss = result["loss"]
        loss.backward()
    print(f"Average Time: {(time.time() - start)/timestep}")
