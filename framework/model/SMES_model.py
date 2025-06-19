# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from abc import ABC, abstractmethod

import einops
import torch
import torch.nn as nn

from .projector import load_mm_projector, build_vision_projector, build_audio_projector
from .encoder import build_vision_tower, build_audio_tower
from ..constants import IGNORE_INDEX, NUM_FRAMES, MODAL_INDEX_MAP
from typing import Optional, Tuple, Union, List
from .mamba_compressor import MambaCompressor

class SMESModel:
    
    def __init__(self, config):
    
        super(SMESModel, self).__init__(config)
        
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.vision_tower.to(device=config.device, dtype=torch.float32)
            self.config.mm_hidden_size = self.vision_tower.hidden_size
            self.mm_projector = build_vision_projector(config)
        if hasattr(config, "mm_audio_tower"):
            self.audio_tower, audio_tower_cfg = build_audio_tower(config, delay_load=True)
            self.audio_tower.to(device=config.device, dtype=torch.float32)
            config.hidden_size_a = config.hidden_size
            self.mm_projector_a = build_audio_projector(config)
        if hasattr(config, "mamba_compressor"):
            self.mamba_compressor = MambaCompressor.from_pretrained(config.mamba_compressor)
            self.mamba_compressor.to(device=config.device, dtype=torch.float32)

            if config.freeze_mamba_compressor:
                for param in self.mamba_compressor.parameters():
                    param.requires_grad = False

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_audio_tower(self):
        audio_tower = getattr(self, 'audio_tower', None)
        if type(audio_tower) is list:
            audio_tower = audio_tower[0]
        return audio_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            if os.path.exists(pretrain_mm_mlp_adapter):
                is_local = True
                if os.path.isdir(pretrain_mm_mlp_adapter):
                    mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)
                else:
                    mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                is_local = False
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.replace('mm_projector.bin', '')
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)


    def initialize_audio_modules(self, model_args, fsdp=None):
        audio_tower = model_args.audio_tower
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter_a
        self.config.mm_audio_tower = audio_tower
        if self.get_audio_tower() is None:
            audio_tower, audio_tower_cfg = build_audio_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.audio_tower = [audio_tower]
            else:
                self.audio_tower = audio_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                audio_tower = self.audio_tower[0]
            else:
                audio_tower = self.audio_tower
        self.config.use_mm_proj = True
        self.config.mm_projector_a_type = getattr(model_args, 'mm_projector_a_type', 'linear')
        self.config.mm_hidden_size_a = audio_tower_cfg.encoder_embed_dim
        self.config.hidden_size_a = audio_tower_cfg.hidden_size
        if getattr(self, 'mm_projector_a', None) is None:
            self.mm_projector_a = build_audio_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector_a.parameters():
                p.requires_grad = True
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector_a.load_state_dict(get_w(mm_projector_weights, 'mm_projector_a'), strict=True)
            

class SMESModelForClassification(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def num_frames(self):
        if hasattr(self.config, 'num_frames'):
            return self.config.num_frames
        else:
            return NUM_FRAMES

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_audio_tower(self):
        return self.get_model().get_audio_tower()

    def encode_images_or_videos(self, images):
        num_frames = self.config.num_frames if hasattr(self.config, 'num_frames') else NUM_FRAMES

        data_batch = []
        for i, (data, modal) in enumerate(images):
            if modal == 'image':
                data = data.expand(num_frames, -1, -1, -1)
            else:
                data = data
            data_batch.append(data)

        data_batch = torch.stack(data_batch, dim=0)

        assert len(data_batch.size()) == 5
        batch_size = data_batch.size(0)

        frames = einops.rearrange(data_batch, 'b t c h w -> (b t) c h w')
        frames_features = self.get_model().get_vision_tower()(frames)
        frames_features = einops.rearrange(frames_features, '(b t) n h -> b t n h', b = batch_size)

        return self.temporal_aggregator(frames_features)

    def temporal_aggregator(self, frames_features):
        """Temporal aggregation of frame features.
        Args:
            frames_features (torch.Tensor): Frame features with shape (b, t, n, h).
        Returns:
            torch.Tensor: Video features with shape (b, n, h).
        """
        # TODO: improve the merging method.
        # *********** mean pooling *************
        if self.config.mm_projector_type == "mlp2x_gelu" or self.config.mm_projector_type == "linear":
            video_features = self.get_model().mm_projector(frames_features.mean(1))
        # *********** spatial convolution *************
        elif self.config.mm_projector_type == "spatial_conv":
            video_features = self.get_model().mm_projector(frames_features)
        # *********** spatial pooling *************
        elif self.config.mm_projector_type == "spatial_pool":
            video_features = self.get_model().mm_projector(frames_features)
        # *********** time  ************
        elif "tc_connector" in self.config.mm_projector_type or "tp_connector" in self.config.mm_projector_type:
            video_features = self.get_model().mm_projector(frames_features)
        else:
            raise Exception(f"Unsupported projector type {self.config.mm_projector_type}!!!")

        return video_features

    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, history_input_ids, 
        k_prev_video_feat=None, k_prev_audio_feat=None
    ):
        vision_tower = self.get_vision_tower()
        audio_tower = self.get_audio_tower()
        
        # NOTE: Handle pure text input
        if (vision_tower is None and audio_tower is None) or images is None or input_ids.shape[1] == 1:
            # if past_key_values is not None and vision_tower is not None and Xs is not None and input_ids.shape[1] == 1:
            #    attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        # Extract multimodal embeddings
        if audio_tower is None:
            mm_features = self.encode_images_or_videos(images)
        elif audio_tower is not None and vision_tower is not None and any(modal == 'video' for (_, modal) in images):
            # [tensor, "image"]
            # [tensor, "audio"]
            # [tensor, "video"]
            # [dict(”, audio), "video"]
            # Handle both video and audio
            X_video, X_audio = [], []
            select_audio_id, select_videoimage_id = [], []

            for idx, data_list in enumerate(images):
                if isinstance(data_list[0], dict):  # Process video-audio pair
                    assert data_list[1] == "video"
                    X_audio.append(data_list[0]["audio"])
                    select_audio_id.append(True)
                    X_video.append((data_list[0]["video"], "video"))
                    select_videoimage_id.append(True)
                else:
                    if data_list[1] == "audio":
                        X_audio.append(data_list[0])
                        select_audio_id.append(True)
                        select_videoimage_id.append(False)
                    elif data_list[1] == "video" or data_list[1] == "image":
                        X_video.append(data_list)
                        select_videoimage_id.append(True)
                        select_audio_id.append(False)

            if len(X_audio) > 0:
               
                Xa_features = torch.cat(X_audio, dim=0)
                audio_padding_mask = torch.zeros(Xa_features.shape, device=self.device).bool()
                audio_embedding, _, _ = self.get_model().get_audio_tower().extract_features(
                    Xa_features, padding_mask=audio_padding_mask, feature_only=True
                )
                Xa_features = self.get_model().mm_projector_a(audio_embedding)
                Xa_features = Xa_features.view(len(X_audio), -1, Xa_features.shape[-1])

            if len(X_video) > 0:
                X_features = self.encode_images_or_videos(X_video)

            
            
            # Combine multimodal embeddings
            mm_features = []
            idx_a, idx_v = 0, 0
            for audio_idx, videoimage_idx in zip(select_audio_id, select_videoimage_id):
                if audio_idx and videoimage_idx:
                    mm_features.append(torch.cat([X_features[idx_v], Xa_features[idx_a]], dim=0))
                    idx_a += 1
                    idx_v += 1
                elif audio_idx:
                    mm_features.append(Xa_features[idx_a])
                    idx_a += 1
                elif videoimage_idx:
                    mm_features.append(X_features[idx_v])
                    idx_v += 1
                else:
                    raise NotImplementedError
                
            
            if k_prev_video_feat is not None and k_prev_audio_feat is not None:
                # The k video and audio features is already concateneated so we only need to pass through the projectors 
                # Each instance contain a list of 5 videos features
                k_prev_video_embs = []
                for video_feats in k_prev_video_feat:  # Iterate through batch instances
                    video_embs = [self.temporal_aggregator(feat) for feat in video_feats]  # Process each video feature in instance
                    k_prev_video_embs.append(torch.cat(video_embs, dim=1))  # Concatenate along feature dimension
                    del video_feats
                k_prev_video_embs = torch.stack(k_prev_video_embs)  # Stack into a batch tensor
                
                del k_prev_video_feat
                torch.cuda.empty_cache()

                
                k_prev_audio_embs = []
                for audio_feats in k_prev_audio_feat:
                    audio_embs = [self.get_model().mm_projector_a(feat) for feat in audio_feats]  # Process each audio feature in instance
                    k_prev_audio_embs.append(torch.cat(audio_embs, dim=1))  # Concatenate along feature dimension
                    del audio_feats
                k_prev_audio_embs = torch.stack(k_prev_audio_embs)  # Stack into a batch tensor
                
                del k_prev_audio_feat
                torch.cuda.empty_cache()

            elif k_prev_video_feat:
                k_prev_video_embs= self.temporal_aggregator(k_prev_video_feat)
            elif k_prev_audio_feat: 
                k_prev_audio_embs= self.get_model().mm_projector_a(k_prev_audio_feat)
            else : 
                k_prev_audio_embs = None 
                k_prev_video_embs = None
            
        
        else:
            data_batch = []
            for i, (data, modal) in enumerate(images):
                data_batch.append(data)
            X_features = torch.cat(data_batch, dim=0)
            audio_padding_mask = torch.zeros(X_features.shape, device=self.device).bool()
            audio_embedding, T, F = self.get_model().get_audio_tower().extract_features(X_features, 
                padding_mask=audio_padding_mask, feature_only=True)      
            mm_features = self.get_model().mm_projector_a(audio_embedding)
            #X_features = X_features.view(len(X_features), -1, X_features.shape[-1])
            

        # Get conversation history representation
        have_hist = history_input_ids is not None
        hist_features = self.get_model().mamba_compressor(history_input_ids) if have_hist else None

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_mm_idx, cur_his_idx = 0, 0
        cur_vid_idx, cur_aud_idx = 0, 0  # Indices for previous video/audio embeddings

        # Iterate through each batch
        for batch_idx, cur_input_ids in enumerate(input_ids):  
            num_multimodals = sum((cur_input_ids == mm_token_idx).sum() for mm_token_idx in MODAL_INDEX_MAP.values())
            # num_prev_vid = (cur_input_ids == MODAL_INDEX_MAP["<kVidHis>"]).sum().item()
            # num_prev_aud = (cur_input_ids == MODAL_INDEX_MAP["<kAudHis>"]).sum().item()

            # Handle text-only input
            if num_multimodals == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_mm_features = mm_features[cur_mm_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_mm_features[0:0], cur_input_embeds_2], dim=0)
                if have_hist:
                    cur_input_embeds = torch.cat([cur_input_embeds, hist_features[cur_his_idx]], dim=0)
                    cur_his_idx += 1
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_mm_idx += 1
                continue

            # Process multimodal tokens
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
          
                # assert cur_labels.shape == cur_input_ids.shape

            mm_token_indices = torch.where(sum([cur_input_ids == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]
            
   
            while mm_token_indices.numel() > 0:
                cur_mm_features = mm_features[cur_mm_idx]
                mm_token_start = mm_token_indices[0]
                # Add tokens before modal token
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:mm_token_start]))  

                # Replace modal token
                if cur_input_ids[mm_token_start] == MODAL_INDEX_MAP['<history>'] and have_hist:
                    cur_new_input_embeds.append(hist_features[cur_his_idx])
                    cur_his_idx += 1
                elif cur_input_ids[mm_token_start] == MODAL_INDEX_MAP['<kVidHis>'] and k_prev_video_embs is not None:
                    cur_new_input_embeds.append(k_prev_video_embs[cur_vid_idx][0])
                    cur_vid_idx += 1
                elif cur_input_ids[mm_token_start] == MODAL_INDEX_MAP['<kAudHis>'] and k_prev_audio_embs is not None:
                    cur_new_input_embeds.append(k_prev_audio_embs[cur_aud_idx][0])
                    cur_aud_idx += 1
                else:
                    cur_new_input_embeds.append(mm_features[cur_mm_idx])
                    cur_mm_idx += 1

                # Mask labels for modal tokens
                if labels is not None:
                    cur_new_labels.append(cur_labels[:mm_token_start])
                    if cur_input_ids[mm_token_start] == MODAL_INDEX_MAP['<history>']:
                        if have_hist:
                            cur_new_labels.append(torch.full((cur_hist_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    else:
                        cur_new_labels.append(torch.full((cur_mm_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[mm_token_start+1:]

                # Update input and indices
                cur_input_ids = cur_input_ids[mm_token_start+1:]
                #Update the list of modal tokens
                mm_token_indices = torch.where(sum([cur_input_ids == mm_token_idx for mm_token_idx in MODAL_INDEX_MAP.values()]))[0]

            # Append remaining input
            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            for x in cur_new_input_embeds:
                print(x.shape)
            new_input_embeds.append(torch.cat(cur_new_input_embeds, dim=0))
            if labels is not None:
                new_labels.append(torch.cat(cur_new_labels, dim=0))
        

        # padding
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels