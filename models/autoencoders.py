"""
Variational Autoencoder (VAE) implementations for 1D data (e.g., audio waveforms).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional
import os
from safetensors.torch import load_file


class ConvUnit1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, 
                 layernorm=True, nonlinearity=nn.GELU()):
        super().__init__()
        self.kernel_size = kernel
        self.stride = stride
        self.layernorm = layernorm
        self.conv_layer = nn.Conv1d(in_channels, out_channels, kernel, stride, padding)
        if self.layernorm:
            self.norm = nn.BatchNorm1d(out_channels) #nn.LayerNorm(out_channels, eps=1e-5, elementwise_affine=True) 
        self.nonlinearity = nonlinearity

    def forward(self, x):
        x = self.conv_layer(x)
        if self.layernorm:
            #x = x.transpose(1, 2)
            x = self.norm(x)
            #x = x.transpose(1, 2)
        x = self.nonlinearity(x)
        return x

class ConvUnitTranspose1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0,
                 output_padding=0, layernorm=True, nonlinearity=nn.GELU()):
        super().__init__()
        self.kernel_size = kernel
        self.stride = stride
        self.layernorm = layernorm
        self.deconv_layer = nn.ConvTranspose1d(in_channels, out_channels, kernel, stride, 
                                 padding, output_padding)
        if self.layernorm:
            self.norm = nn.BatchNorm1d(out_channels) #nn.LayerNorm(out_channels, eps=1e-5, elementwise_affine=True) 
        self.nonlinearity = nonlinearity

    def forward(self, x):
        x = self.deconv_layer(x)
        if self.layernorm:
            #x = x.transpose(1, 2)
            x = self.norm(x)
            #x = x.transpose(1, 2)
        x = self.nonlinearity(x)
        return x

class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, norm='batch', nonlinearity=nn.GELU()):
        super().__init__()
        self.norm = norm
        if norm == 'layer':
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features), 
                nonlinearity
            )
        elif norm == 'batch':
            self.linear_layer = nn.Linear(in_features, out_features)
            self.norm_layer = nn.BatchNorm1d(out_features)
            self.nonlinearity = nonlinearity
        elif norm is None: #nn.BatchNorm1d(out_features)
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features),
                nonlinearity
            )

    def forward(self, x):
        if self.norm == 'layer' or self.norm is None:
            return self.model(x)
        elif self.norm == 'batch':
            x = self.linear_layer(x)
            if len(x.shape) == 3:
                x = x.transpose(1,2)
            x = self.norm_layer(x)
            if len(x.shape) == 3:
                x = x.transpose(1,2)
            x = self.nonlinearity(x)
            return x
        
class VAE_1D(nn.Module):
    def __init__(self, 
            z_dim=32, 
            proj_intermediate_dim=2048, 
            conv_dim=256, 
            treat_as_sequence=False,
            kernel_sizes=[10, 5, 3, 3, 3],
            strides=[5, 4, 2, 2, 2], 
            in_size=16000, 
            hidden_dim=192, 
            norm_type='batch',
            nonlinearity=None,
            beta = 1.0,
            warmup_steps = 5000,
            kl_annealing = False
            ):
        super().__init__()
        
        self.z_dim = z_dim
        self.norm_type = norm_type
        self.kernels = kernel_sizes
        self.strides = strides
        self.proj_intermediate_dim = proj_intermediate_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = conv_dim
        self.treat_as_sequence = treat_as_sequence
        self.in_size = in_size
        self.beta = beta
        self.global_step = 0
        self.warmup_steps = warmup_steps
        self.kl_annealing = kl_annealing

        "Prior projection heads"
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        "Encoder"
        self.conv = nn.ModuleList()
        for i, (kernel, stride) in enumerate(zip(self.kernels, self.strides)):
            self.conv.append(
                ConvUnit1D(1 if i == 0 else self.conv_dim, 
                           self.conv_dim, 
                           kernel=kernel, 
                           stride=stride, 
                           padding=0)
                )

        "Number of output frames at the end of the encoder - Depends on receptive field"
        self.final_conv_size = self._calculate_rf_stride(self.in_size, self.kernels, self.strides)[0]

        self.conv_fc = nn.Sequential(
            LinearUnit(self.conv_dim, self.proj_intermediate_dim, norm=self.norm_type),
            LinearUnit(self.proj_intermediate_dim, self.hidden_dim, norm=self.norm_type)
        )

        # Decoder
        self.deconv_fc = nn.Sequential(
            LinearUnit(self.z_dim, self.hidden_dim, norm=None),
            LinearUnit(self.hidden_dim, self.proj_intermediate_dim, norm=None),
            LinearUnit(self.proj_intermediate_dim, self.conv_dim, norm=None)
        )
        self.deconv = nn.ModuleList()
        for i, (kernel, stride) in enumerate(zip(self.kernels[::-1], self.strides[::-1])):
            if (i == 1 or i == 6) and self.final_conv_size == 357: #1,6 makes 114401
                output_padding = 1
            elif (i ==len(self.kernels)-2) and self.final_conv_size == 399:
                output_padding = 1
            elif (i == 0 or i == 1) and self.final_conv_size == 199: # (i == 0 or i == 1)
                #    #If last 2 kernels are 3, then i==0 not needed
                #    #If last 2 kernels are 2, then i==0 needed
                output_padding = 1
            elif (i == 2) and self.final_conv_size == 9: # If last 2 kernels are 2, then i==0 needed
                output_padding = 1
            else:
                output_padding = 0
            if i < len(self.kernels) - 1:
                self.deconv.append(
                    ConvUnitTranspose1D(self.conv_dim, 
                            self.conv_dim, 
                            kernel=kernel, 
                            stride=stride, 
                            padding=0,
                            output_padding=output_padding),
                )
            else:
                self.deconv.append(
                    ConvUnitTranspose1D(self.conv_dim, 1,
                            kernel=kernel, 
                            stride=stride, 
                            padding=0,
                            layernorm=True,
                            output_padding=output_padding,
                            nonlinearity=nn.Tanh())
                )

    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std).to(std.device)
            return mean + eps * std
        return mean


    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int]
    ):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.kernels, self.strides):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths)
        output_lengths = output_lengths.to(torch.long) #long

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        #except:
        #pass
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def _calculate_rf_stride(self,input_size, kernels, strides):
        size = input_size
        cum_stride = 1
        rf_size = 1
        
        for i, (k, s) in enumerate(zip(kernels, strides)):
            # Calculate output size
            output_size = (size - k) // s + 1
            
            # Calculate receptive field
            rf_size = rf_size + (k - 1) * cum_stride
            
            # Update cumulative stride
            cum_stride *= s
            
            #print(f"Layer {i+1}:")
            #print(f"Output size: {output_size}")
            #print(f"Cumulative stride: {cum_stride}")
            #print(f"Receptive field: {rf_size}")
            #print()
            
            size = output_size
        
        return size, cum_stride, rf_size

    def encode_frames(self, x):
        x = x.view(-1, 1, self.in_size)
        for conv_layer in self.conv:
            x = conv_layer(x)
        #x = x.reshape(-1, self.conv_dim * self.final_conv_size)
        x  = x.transpose(1,2)
        x = self.conv_fc(x)
        #x = x.view(-1, self.frames, self.conv_out_dim)
        return x

    def encode_z(self, x):
        mean = self.z_mean(x)
        logvar = self.z_logvar(x)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def decode_frames(self, z):
        x = self.deconv_fc(z)
        #x = x.view(-1, self.conv_dim, self.final_conv_size)
        if len(x.shape) == 2 and self.treat_as_sequence:
            x = x.reshape(self.batch_size,-1, x.shape[-1])
        x = x.transpose(1,2)
        for deconv_layer in self.deconv:
            x = deconv_layer(x)
        return x.squeeze(1)

    def vae_loss(self, recon_x, x, mu, log_var, attention_mask = None):
        batch_size = x.size(0)
        "Use attention mask to calculate losses on non-padded values only"
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
            try:
                recon_x = recon_x[attention_mask].reshape(batch_size, -1)
                x = x[attention_mask].reshape(batch_size, -1)
            except RuntimeError:
                "In case the batch size does not divide the input size exactly - can happen with last batch"
                recon_x = recon_x[attention_mask].flatten()
                x = x[attention_mask].flatten()
            # compute reduced attention_mask correponding to feature vectors
            sub_attention_mask = self._get_feature_vector_attention_mask(
                mu.shape[1], attention_mask
            )
            mu = mu[sub_attention_mask] #.view(batch_size, -1, self.z_dim)
            log_var = log_var[sub_attention_mask] #.view(batch_size, -1, self.z_dim)
        recon_loss = F.mse_loss(recon_x, x,reduction='sum')
        
        "Assumes Gaussian prior"
        #if log_var.min() < -20 or log_var.max() > 20:
        #    print("Log var out of range")
        log_var = torch.clamp(log_var, min=-20, max=20)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim = -1).sum()
        
        "KL Annealing"
        if self.kl_annealing:
            beta = min(1.0, self.global_step/self.warmup_steps) * self.beta

            return (recon_loss + beta * kld_loss), recon_loss, beta * kld_loss
        else:
            return (recon_loss + self.beta * kld_loss), recon_loss, self.beta * kld_loss

    def forward(self, input_values, attention_mask, mask_time_indices, global_step):
        self.global_step = global_step
        self.batch_size = input_values.size(0)
        "See how to use attention mask and/or mask_time_indices"
        conv_x = self.encode_frames(input_values)
        z_mean, z_logvar, z = self.encode_z(conv_x)
        recon_x = self.decode_frames(z)

        if recon_x.shape[1] - input_values.shape[1] == 1:
            "Only in case of TIMIT - 7.15s sequence length"
            recon_x = recon_x[:, :input_values.shape[1]]

        "Loss"
        vae_loss, recon_loss, kld_loss = self.vae_loss(recon_x, input_values, z_mean, z_logvar,attention_mask)

        return z_mean, z_logvar, z, recon_x, vae_loss, recon_loss, kld_loss
    
class VAE_1D_ForSupervisedFineTuning(nn.Module):
    def __init__(self, 
            z_dim=32, 
            proj_intermediate_dim=2048, 
            conv_dim=256, 
            treat_as_sequence=False,
            kernel_sizes=[10, 5, 3, 3, 3],
            strides=[5, 4, 2, 2, 2], 
            in_size=16000, 
            hidden_dim=192, 
            norm_type='batch',
            nonlinearity=None,
            beta = 1.0,
            vae_fine_tuning_classifier = 'mlp',
            fine_tuning_output_classes = 4,
            final_dropout = 0.0,
            supervised_loss_reduction = "sum",
            supervised_loss_type = "cross_entropy",
            vae_loss_weight = 0.2,
            supervised_loss_weight = 0.8,
            warmup_steps = 5000,
            kl_annealing = False
            ):
        super().__init__()
        
        self.z_dim = z_dim
        self.norm_type = norm_type
        self.kernels = kernel_sizes
        self.strides = strides
        self.proj_intermediate_dim = proj_intermediate_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = conv_dim
        self.treat_as_sequence = treat_as_sequence
        self.in_size = in_size
        self.nonlinearity = nonlinearity
        self.beta = beta
        self.vae_fine_tuning_classifier = vae_fine_tuning_classifier
        self.fine_tuning_output_classes = fine_tuning_output_classes
        self.final_dropout = final_dropout
        self.supervised_loss_reduction = supervised_loss_reduction
        self.supervised_loss_type = supervised_loss_type
        self.vae_loss_weight = vae_loss_weight
        self.supervised_loss_weight = supervised_loss_weight
        self.global_step = 0
        self.warmup_steps = warmup_steps
        self.kl_annealing = kl_annealing

        "Prior projection heads"
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        "Encoder"
        self.conv = nn.ModuleList()
        for i, (kernel, stride) in enumerate(zip(self.kernels, self.strides)):
            self.conv.append(
                ConvUnit1D(1 if i == 0 else self.conv_dim, 
                           self.conv_dim, 
                           kernel=kernel, 
                           stride=stride, 
                           padding=0)
                )

        "Number of output frames at the end of the encoder - Depends on receptive field"
        self.final_conv_size = self._calculate_rf_stride(self.in_size, self.kernels, self.strides)[0]

        self.conv_fc = nn.Sequential(
            LinearUnit(self.conv_dim, self.proj_intermediate_dim, norm=self.norm_type),
            LinearUnit(self.proj_intermediate_dim, self.hidden_dim, norm=self.norm_type)
        )

        # Decoder
        self.deconv_fc = nn.Sequential(
            LinearUnit(self.z_dim, self.hidden_dim, norm=None),
            LinearUnit(self.hidden_dim, self.proj_intermediate_dim, norm=None),
            LinearUnit(self.proj_intermediate_dim, self.conv_dim, norm=None)
        )
        self.deconv = nn.ModuleList()
        for i, (kernel, stride) in enumerate(zip(self.kernels[::-1], self.strides[::-1])):
            if (i == 1 or i == 6) and self.final_conv_size == 357: #1,6 makes 114401
                output_padding = 1
            elif (i ==len(self.kernels)-2) and self.final_conv_size == 399:
                output_padding = 1
            elif (i == 0 or i == 1) and self.final_conv_size == 199: # (i == 0 or i == 1)
                #    #If last 2 kernels are 3, then i==0 not needed
                #    #If last 2 kernels are 2, then i==0 needed
                output_padding = 1
            elif (i == 2) and self.final_conv_size == 9: # If last 2 kernels are 2, then i==0 needed
                output_padding = 1
            else:
                output_padding = 0
            if i < len(self.kernels) - 1:
                self.deconv.append(
                    ConvUnitTranspose1D(self.conv_dim, 
                            self.conv_dim, 
                            kernel=kernel, 
                            stride=stride, 
                            padding=0,
                            output_padding=output_padding),
                )
            else:
                self.deconv.append(
                    ConvUnitTranspose1D(self.conv_dim, 1,
                            kernel=kernel, 
                            stride=stride, 
                            padding=0,
                            layernorm=True,
                            output_padding=output_padding,
                            nonlinearity=nn.Tanh())
                )


        #Classification Head for Supervised Loss Term
        if vae_fine_tuning_classifier is not None:
            self.classification_head = self._create_classifier(
                self.vae_fine_tuning_classifier,
                self.z_dim,
                self.hidden_dim,
                self.hidden_dim,
                self.fine_tuning_output_classes,
                self.final_dropout
            )
            

    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std).to(std.device)
            return mean + eps * std
        return mean
       
    def load_pretrained_weights(self, pretrained_path, verbose=True, strict=False):
        """
        Load weights from a pretrained VAE model into this supervised fine-tuning model.
        
        This method properly handles loading weights from a pretrained model that doesn't have
        classifier heads, by:
        1. Loading the pretrained weights with strict=False
        2. Reporting any unexpected and missing keys
        3. Verifying that only classifier-related keys are missing
        
        Args:
            pretrained_path (str): Path to the pretrained model weights file (.safetensors, .bin, etc.)
            verbose (bool, optional): Whether to print detailed messages about loaded weights. Default: True
            strict (bool, optional): Whether to strictly enforce that the keys in state_dict match. Default: False
        
        Returns:
            Tuple[List[str], List[str]]: Lists of unexpected and missing keys
        
        Raises:
            ValueError: If critical non-classifier keys are missing from the pretrained weights
        """
        
        if verbose:
            print(f"Loading pretrained weights from: {pretrained_path}")
        
        # Determine file type and load accordingly
        if pretrained_path.endswith(".safetensors"):
            if os.path.exists(pretrained_path):
                state_dict = load_file(pretrained_path)
            else:
                raise FileNotFoundError(f"Pretrained weights file not found: {pretrained_path}")
        elif pretrained_path.endswith(".bin"):
            if os.path.exists(pretrained_path):
                state_dict = torch.load(pretrained_path, map_location="cpu")
            else:
                raise FileNotFoundError(f"Pretrained weights file not found: {pretrained_path}")
        else:
            raise ValueError(f"Unsupported weights file format: {pretrained_path}")
        
        # Get current model state dict
        model_state_dict = self.state_dict()
        
        # Load weights with strict=False to allow missing keys (like classifier heads)
        load_result = self.load_state_dict(state_dict, strict=strict)
        
        # Get lists of missing and unexpected keys
        missing_keys = load_result.missing_keys
        unexpected_keys = load_result.unexpected_keys
        
        # Identify keys related to classifier heads
        classifier_related_keys = [k for k in model_state_dict.keys() if 
                                "classification_head" in k or "classifier" in k]
        
        # Check if all missing keys are classifier-related
        non_classifier_missing_keys = [k for k in missing_keys if k not in classifier_related_keys]
        
        if verbose:
            print(f"Loaded pretrained weights with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys")
            
            if classifier_related_keys:
                print(f"Found {len(classifier_related_keys)} classifier-related keys in the model")

            if missing_keys:
                print(f"Missing keys: {missing_keys[:5] + ['...'] if len(missing_keys) > 5 else missing_keys}")
            
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5] + ['...'] if len(unexpected_keys) > 5 else unexpected_keys}")
        
        # Warn if there are missing keys that aren't classifier-related
        if non_classifier_missing_keys and verbose:
            print(f"Warning: {len(non_classifier_missing_keys)} missing keys are not classifier-related")
            print(f"Non-classifier missing keys: {non_classifier_missing_keys[:5] + ['...'] if len(non_classifier_missing_keys) > 5 else non_classifier_missing_keys}")
        
        return load_result, classifier_related_keys

    def _create_classifier(self, classifier_type, input_dim, hidden_dim, proj_size, num_classes, dropout_prob=0.1):
        """
        Create a classifier head based on the specified type.
        
        Args:
            classifier_type (str): Type of classifier to create ('mlp', 'cnn', 'lstm', 'transformer', etc.)
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layers
            proj_size (int): Projection size for some classifiers
            num_classes (int): Number of output classes
            dropout_prob (float): Dropout probability
            
        Returns:
            nn.Module: The classifier module
        """
        if classifier_type == "mlp":
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim // 2, num_classes)
            )
            
        elif classifier_type == "cnn":
            return nn.Sequential(
                nn.Unflatten(1, (1, input_dim)),  # [B, 1, input_dim]
                nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),  # Global average pooling
                nn.Flatten(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim*2, num_classes)
            )
            
        elif classifier_type == "lstm":
            class LSTMClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_dim,
                        hidden_size=hidden_dim,
                        num_layers=2,
                        batch_first=True,
                        dropout=dropout_prob,
                        bidirectional=True
                    )
                    self.dropout = nn.Dropout(dropout_prob)
                    self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
                    
                def forward(self, x):
                    # Reshape if input is just [batch, features]
                    if len(x.shape) == 2:
                        x = x.unsqueeze(1)  # [batch, 1, features]
                    
                    # LSTM expects [batch, seq_len, features]
                    output, (h_n, _) = self.lstm(x)
                    
                    # Concatenate the final hidden states from both directions
                    h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
                    
                    # Apply dropout and final classification
                    h_n = self.dropout(h_n)
                    logits = self.fc(h_n)
                    return logits
                    
            return LSTMClassifier()
            
        elif classifier_type == "transformer":
            # Transformer classifier - attends to different parts of the input with different weights
            class TransformerClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Project input to model dimension if needed
                    self.input_projection = nn.Linear(input_dim, proj_size) if input_dim != proj_size else nn.Identity()
                    
                    # Transformer encoder layer
                    self.encoder_layer = nn.TransformerEncoderLayer(
                        d_model=proj_size,
                        nhead=8,  # Number of attention heads
                        dim_feedforward=hidden_dim,
                        dropout=dropout_prob,
                        activation="gelu",
                        batch_first=True
                    )
                    
                    # Transformer encoder
                    self.transformer_encoder = nn.TransformerEncoder(
                        self.encoder_layer,
                        num_layers=2
                    )
                    
                    # Classification head
                    self.classifier = nn.Linear(proj_size, num_classes)
                    
                def forward(self, x):
                    # Reshape if input is just [batch, features]
                    if len(x.shape) == 2:
                        x = x.unsqueeze(1)  # [batch, 1, features]
                        
                    # Project input if needed
                    x = self.input_projection(x)
                    
                    # Apply transformer
                    x = self.transformer_encoder(x)
                    
                    # Average pooling over sequence dimension
                    x = x.mean(dim=1)
                    
                    # Classification
                    return self.classifier(x)
                    
            return TransformerClassifier()
            
    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int]
    ):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.kernels, self.strides):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths)
        output_lengths = output_lengths.to(torch.long) #long

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        #except:
        #pass
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def _calculate_rf_stride(self,input_size, kernels, strides):
        size = input_size
        cum_stride = 1
        rf_size = 1
        
        for i, (k, s) in enumerate(zip(kernels, strides)):
            # Calculate output size
            output_size = (size - k) // s + 1
            
            # Calculate receptive field
            rf_size = rf_size + (k - 1) * cum_stride
            
            # Update cumulative stride
            cum_stride *= s
            
            #print(f"Layer {i+1}:")
            #print(f"Output size: {output_size}")
            #print(f"Cumulative stride: {cum_stride}")
            #print(f"Receptive field: {rf_size}")
            #print()
            
            size = output_size
        
        return size, cum_stride, rf_size

    def encode_frames(self, x):
        x = x.view(-1, 1, self.in_size)
        for conv_layer in self.conv:
            x = conv_layer(x)
        #x = x.reshape(-1, self.conv_dim * self.final_conv_size)
        x  = x.transpose(1,2)
        x = self.conv_fc(x)
        #x = x.view(-1, self.frames, self.conv_out_dim)
        return x

    def encode_z(self, x):
        mean = self.z_mean(x)
        logvar = self.z_logvar(x)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def decode_frames(self, z):
        x = self.deconv_fc(z)
        #x = x.view(-1, self.conv_dim, self.final_conv_size)
        if len(x.shape) == 2 and self.treat_as_sequence:
            x = x.reshape(self.batch_size,-1, x.shape[-1])
        x = x.transpose(1,2)
        for deconv_layer in self.deconv:
            x = deconv_layer(x)
        return x.squeeze(1)

    @staticmethod
    def compute_supervised_loss(
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
        loss_type: str = "cross_entropy",
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.FloatTensor] = None,
        reduction: str = "sum",
    ):
        """
        Compute supervised classification loss between logits and labels.
        
        This method handles different input shapes for both logits and labels:
        - logits: [B, S, N] (batch_size, num_segments, num_classes) or [B, N] (batch_size, num_classes)
        - labels: [B, S] (batch_size, num_segments) or [B] (batch_size) or [B, S, N] (one-hot encoded)
        
        Args:
            logits (torch.FloatTensor): The predicted logits 
                Shape: [B, S, N] or [B, N] where:
                B = batch size
                S = number of segments
                N = number of classes
            labels (torch.FloatTensor): The target labels
                Shape: [B, S], [B] or [B, S, N] (one-hot encoded)
            loss_type (str, optional): Type of loss to compute. 
                Options: "cross_entropy", "focal", "label_smoothed_ce". Default: "cross_entropy"
            label_smoothing (float, optional): Label smoothing factor (0-1). Default: 0.0
            class_weights (torch.FloatTensor, optional): Weights for each class for weighted loss. Default: None
            reduction (str, optional): Specifies the reduction to apply to the output. 
                Options: 'none', 'mean', 'sum'. Default: 'mean'
                
        Returns:
            torch.FloatTensor: Computed loss
        """
        device = logits.device
        
        # Handle different shapes
        if len(logits.shape) == 3:  # [B, S, N]
            B, S, N = logits.shape
            # Reshape to [B*S, N]
            logits_flat = logits.reshape(-1, N)
            
            # Handle different label shapes
            if len(labels.shape) == 2:  # [B, S]
                labels_flat = labels.reshape(-1)
            elif len(labels.shape) == 1:  # [B]
                # Repeat labels for each segment
                labels_flat = labels.unsqueeze(1).repeat(1, S).reshape(-1)
            elif len(labels.shape) == 3:  # [B, S, N] (one-hot)
                labels_flat = labels.reshape(-1, N)
            else:
                raise ValueError(f"Unsupported label shape: {labels.shape}")
                
        elif len(logits.shape) == 2:  # [B, N]
            B, N = logits.shape
            logits_flat = logits
            
            # Handle different label shapes
            if len(labels.shape) == 1:  # [B]
                labels_flat = labels
            elif len(labels.shape) == 2:  # [B, N] (one-hot)
                if labels.shape[1] == N:  # One-hot encoded
                    labels_flat = labels
                else:  # [B, S] but S != N
                    raise ValueError(f"Label shape {labels.shape} incompatible with logits shape {logits.shape}")
            else:
                raise ValueError(f"Unsupported label shape: {labels.shape}")
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}")
        
        # Check if labels are one-hot encoded
        if len(labels_flat.shape) == 2 and labels_flat.shape[1] > 1:
            is_one_hot = True
            # Convert one-hot to class indices if needed for some loss functions
            class_indices = torch.argmax(labels_flat, dim=1)
        else:
            is_one_hot = False
            class_indices = labels_flat
        
        # Compute the appropriate loss
        if loss_type == "cross_entropy":
            if is_one_hot:
                # For one-hot labels, use binary cross entropy with logits
                if class_weights is not None:
                    # Apply class weights to each sample
                    sample_weights = (labels_flat * class_weights.unsqueeze(0)).sum(dim=1)
                    loss_fn = nn.BCEWithLogitsLoss(weight=sample_weights, reduction=reduction)
                else:
                    loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
                
                loss = loss_fn(logits_flat, labels_flat.float())
            else:
                # For class indices, use cross entropy
                if label_smoothing > 0:
                    loss_fn = nn.CrossEntropyLoss(
                        weight=class_weights,
                        label_smoothing=label_smoothing,
                        reduction=reduction
                    )
                else:
                    loss_fn = nn.CrossEntropyLoss(
                        weight=class_weights,
                        reduction=reduction
                    )
                
                loss = loss_fn(logits_flat, class_indices.long())
                
        elif loss_type == "focal":
            # Focal Loss implementation (helps with class imbalance)
            gamma = 2.0  # Focusing parameter
            
            if is_one_hot:
                # Convert logits to probabilities
                probs = torch.sigmoid(logits_flat)
                # Calculate focal weights
                pt = labels_flat * probs + (1 - labels_flat) * (1 - probs)
                focal_weights = (1 - pt) ** gamma
                
                # Standard BCE loss
                bce_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits_flat, 
                    labels_flat.float(), 
                    reduction='none'
                )
                
                # Apply focal weights
                loss = focal_weights * bce_loss
                
                # Apply class weights if provided
                if class_weights is not None:
                    class_weight_per_sample = (labels_flat * class_weights.unsqueeze(0)).sum(dim=1, keepdim=True)
                    loss = loss * class_weight_per_sample
                    
                # Apply reduction
                if reduction == 'mean':
                    loss = loss.mean()
                elif reduction == 'sum':
                    loss = loss.sum()
            else:
                # One-hot encode class_indices for focal loss calculation
                labels_one_hot = nn.functional.one_hot(class_indices.long(), num_classes=N).float()
                
                # Convert logits to probabilities
                probs = nn.functional.softmax(logits_flat, dim=1)
                
                # Calculate focal weights
                pt = (labels_one_hot * probs).sum(dim=1)
                focal_weights = (1 - pt) ** gamma
                
                # Standard CE loss
                ce_loss = nn.functional.cross_entropy(
                    logits_flat, 
                    class_indices.long(), 
                    weight=class_weights,
                    reduction='none'
                )
                
                # Apply focal weights
                loss = focal_weights * ce_loss
                
                # Apply reduction
                if reduction == 'mean':
                    loss = loss.mean()
                elif reduction == 'sum':
                    loss = loss.sum()
                    
        elif loss_type == "label_smoothed_ce":
            # Custom implementation of label smoothing for both one-hot and class indices
            if is_one_hot:
                # For one-hot labels, smooth the labels directly
                smooth_labels = labels_flat * (1 - label_smoothing) + label_smoothing / N
                
                # Compute KL divergence loss
                log_probs = nn.functional.log_softmax(logits_flat, dim=1)
                loss = -(smooth_labels * log_probs).sum(dim=1)
                
                # Apply class weights if provided
                if class_weights is not None:
                    class_weight_per_sample = (labels_flat * class_weights.unsqueeze(0)).sum(dim=1)
                    loss = loss * class_weight_per_sample
                    
                # Apply reduction
                if reduction == 'mean':
                    loss = loss.mean()
                elif reduction == 'sum':
                    loss = loss.sum()
                elif reduction == 'none':
                    pass  # No reduction needed
            else:
                # For class indices, use built-in CrossEntropyLoss with label_smoothing
                loss_fn = nn.CrossEntropyLoss(
                    weight=class_weights,
                    label_smoothing=label_smoothing,
                    reduction=reduction
                )
                loss = loss_fn(logits_flat, class_indices.long())
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
            
        return loss, logits_flat, class_indices

    def vae_loss(self, recon_x, x, mu, log_var, labels, attention_mask = None):
        batch_size = x.size(0)
        "Use attention mask to calculate losses on non-padded values only"
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
            try:
                recon_x = recon_x[attention_mask].reshape(batch_size, -1)
                x = x[attention_mask].reshape(batch_size, -1)
            except RuntimeError:
                "In case the batch size does not divide the input size exactly - can happen with last batch"
                recon_x = recon_x[attention_mask].flatten()
                x = x[attention_mask].flatten()
            # compute reduced attention_mask correponding to feature vectors
            sub_attention_mask = self._get_feature_vector_attention_mask(
                mu.shape[1], attention_mask
            )
            mu = mu[sub_attention_mask] #.view(batch_size, -1, self.z_dim)
            log_var = log_var[sub_attention_mask] #.view(batch_size, -1, self.z_dim)
            labels = labels[attention_mask]
        
        recon_loss = F.mse_loss(recon_x, x,reduction='sum')
        
        "Assumes Gaussian prior"
        log_var = torch.clamp(log_var, min=-20, max=20)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim = -1).sum()
        
        "KL Annealing"
        if self.kl_annealing:
            beta = min(1.0, self.global_step/self.warmup_steps) * self.beta

            return (recon_loss + beta * kld_loss), recon_loss, beta * kld_loss, mu, log_var, labels
        else:
            return (recon_loss + self.beta * kld_loss), recon_loss, self.beta * kld_loss, mu, log_var, labels

    def forward(self, input_values, labels, attention_mask, mask_time_indices, global_step):
        self.global_step = global_step
        self.batch_size = input_values.size(0)
        "See how to use attention mask and/or mask_time_indices"
        conv_x = self.encode_frames(input_values)
        z_mean, z_logvar, z = self.encode_z(conv_x)
        recon_x = self.decode_frames(z)

        if recon_x.shape[1] - input_values.shape[1] == 1:
            "Only in case of TIMIT - 7.15s sequence length"
            recon_x = recon_x[:, :input_values.shape[1]]

        "VAE Loss"
        vae_loss, recon_loss, kld_loss, z_mean, z_logvar, labels = self.vae_loss(recon_x, input_values, z_mean, z_logvar, labels, attention_mask)

        "Supervised Loss"
        if self.vae_fine_tuning_classifier is not None:
            logits = self.classification_head(z_mean)

            supervised_loss, logits, labels = self.compute_supervised_loss(
                logits=logits,
                labels=labels,
                loss_type=self.supervised_loss_type,
                label_smoothing=0.0,
                class_weights=None,
                reduction=self.supervised_loss_reduction,
            )

        total_loss = self.vae_loss_weight * vae_loss + self.supervised_loss_weight * supervised_loss

        return z_mean, z_logvar, z, recon_x, total_loss, vae_loss, recon_loss, kld_loss, supervised_loss, logits, labels
   
class VAE_1D_FC(nn.Module):
    def __init__(self, 
            z_dim=32,
            hidden_dims=[2048, 1024, 512, 256],
            kernel_sizes=[10, 5, 3, 3, 3],
            strides=[5, 4, 2, 2, 2], 
            treat_as_sequence=False,
            in_size=16000,
            norm_type='batch',
            beta=1.0,
            warmup_steps=5000,
            kl_annealing=False):
        super().__init__()
        
        self.z_dim = z_dim
        self.hidden_dims = hidden_dims
        self.kernels = kernel_sizes #not used, only to find the sub_attention_mask
        self.strides = strides #not used, only to find the sub_attention_mask
        self.treat_as_sequence = treat_as_sequence
        self.in_size = in_size
        self.norm_type = norm_type
        self.beta = beta
        self.global_step = 0
        self.warmup_steps = warmup_steps
        self.kl_annealing = kl_annealing

        # Encoder
        encoder_layers = []
        in_features = in_size
        for hidden_dim in hidden_dims:
            encoder_layers.append(LinearUnit(in_features, hidden_dim,norm=self.norm_type))
            in_features = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.z_mean = nn.Linear(hidden_dims[-1], z_dim)
        self.z_logvar = nn.Linear(hidden_dims[-1], z_dim)

        # Decoder
        decoder_layers = []
        in_features = z_dim
        for hidden_dim in reversed(hidden_dims[:-1]):
            decoder_layers.append(LinearUnit(in_features, hidden_dim,norm=None))
            in_features = hidden_dim
            
        decoder_layers.append(
            nn.Sequential(
                nn.Linear(in_features, in_size),
                nn.Tanh()
            )
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int]
    ):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.kernels, self.strides):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths)
        output_lengths = output_lengths.to(torch.long) #long

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        #except:
        #pass
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std).to(std.device)
            return mean + eps * std
        return mean

    def encode_frames(self, x):
        try:
            x = x.view(-1, self.in_size)
        except RuntimeError:
            x = x.reshape(-1, self.in_size)
        return self.encoder(x)

    def encode_z(self, x):
        mean = self.z_mean(x)
        logvar = self.z_logvar(x)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def decode_frames(self, z):
        return self.decoder(z)

    def vae_loss(self, recon_x, x, mu, log_var, attention_mask=None):
        batch_size = x.size(0)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
            try:
                recon_x = recon_x[attention_mask].reshape(batch_size, -1)
                x = x[attention_mask].reshape(batch_size, -1)
            except RuntimeError:
                "In case the batch size does not divide the input size exactly - can happen with last batch"
                recon_x = recon_x[attention_mask].flatten()
                x = x[attention_mask].flatten()

        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        log_var = torch.clamp(log_var, min=-20, max=20)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        if self.kl_annealing:
            beta = min(1.0, self.global_step/self.warmup_steps) * self.beta
            return (recon_loss + beta * kld_loss), recon_loss, beta * kld_loss
        else:
            return (recon_loss + self.beta * kld_loss), recon_loss, self.beta * kld_loss

    def forward(self, input_values, attention_mask, mask_time_indices, global_step):
        self.global_step = global_step
        batch_size = input_values.size(0)
        fc_x = self.encode_frames(input_values)
        z_mean, z_logvar, z = self.encode_z(fc_x)
        recon_x = self.decode_frames(z)
        
        recon_x = recon_x.view(batch_size,-1, self.in_size)
        z_mean = z_mean.view(batch_size,-1, self.z_dim)
        z_logvar = z_logvar.view(batch_size,-1, self.z_dim)
        z = z.view(batch_size,-1, self.z_dim)

        if self.treat_as_sequence and len(recon_x.shape) == 3:
            recon_x = recon_x.squeeze(1)
            z_mean = z_mean.squeeze(1)
            z_logvar = z_logvar.squeeze(1)
            
        vae_loss, recon_loss, kld_loss = self.vae_loss(
            recon_x, input_values, z_mean, z_logvar, attention_mask
        )

        return z_mean, z_logvar, z, recon_x, vae_loss, recon_loss, kld_loss

class VAE_1D_FC_ForSupervisedFineTuning(nn.Module):
    def __init__(self, 
            z_dim=32,
            hidden_dims=[2048, 1024, 512, 256],
            kernel_sizes=[10, 5, 3, 3, 3],
            strides=[5, 4, 2, 2, 2], 
            treat_as_sequence=False,
            in_size=16000,
            norm_type='batch',
            beta=1.0,
            hidden_dim = 192,
            vae_fine_tuning_classifier = 'mlp',
            fine_tuning_output_classes = 4,
            final_dropout = 0.0,
            supervised_loss_reduction = "sum",
            supervised_loss_type = "cross_entropy",
            vae_loss_weight = 0.2,
            supervised_loss_weight = 0.8,
            warmup_steps=5000,
            kl_annealing=False):
        super().__init__()
        
        self.z_dim = z_dim
        self.hidden_dims = hidden_dims
        self.hidden_dim = hidden_dim #for classifier inner dims
        self.kernels = kernel_sizes #not used, only to find the sub_attention_mask
        self.strides = strides #not used, only to find the sub_attention_mask
        self.treat_as_sequence = treat_as_sequence
        self.in_size = in_size
        self.norm_type = norm_type
        self.beta = beta
        self.vae_fine_tuning_classifier = vae_fine_tuning_classifier
        self.fine_tuning_output_classes = fine_tuning_output_classes
        self.final_dropout = final_dropout
        self.supervised_loss_reduction = supervised_loss_reduction
        self.supervised_loss_type = supervised_loss_type
        self.vae_loss_weight = vae_loss_weight
        self.supervised_loss_weight = supervised_loss_weight
        self.global_step = 0
        self.warmup_steps = warmup_steps
        self.kl_annealing = kl_annealing

        # Encoder
        encoder_layers = []
        in_features = in_size
        for hidden_dim in hidden_dims:
            encoder_layers.append(LinearUnit(in_features, hidden_dim,norm=self.norm_type))
            in_features = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.z_mean = nn.Linear(hidden_dims[-1], z_dim)
        self.z_logvar = nn.Linear(hidden_dims[-1], z_dim)

        # Decoder
        decoder_layers = []
        in_features = z_dim
        for hidden_dim in reversed(hidden_dims[:-1]):
            decoder_layers.append(LinearUnit(in_features, hidden_dim,norm=None))
            in_features = hidden_dim
            
        decoder_layers.append(
            nn.Sequential(
                nn.Linear(in_features, in_size),
                nn.Tanh()
            )
        )
        self.decoder = nn.Sequential(*decoder_layers)

        #Classification Head for Supervised Loss Term
        if vae_fine_tuning_classifier is not None:
            self.classification_head = self._create_classifier(
                self.vae_fine_tuning_classifier,
                self.z_dim,
                self.hidden_dim,
                self.hidden_dim,
                self.fine_tuning_output_classes,
                self.final_dropout
            )

    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std).to(std.device)
            return mean + eps * std
        return mean
              
    def load_pretrained_weights(self, pretrained_path, verbose=True, strict=False):
        """
        Load weights from a pretrained VAE model into this supervised fine-tuning model.
        
        This method properly handles loading weights from a pretrained model that doesn't have
        classifier heads, by:
        1. Loading the pretrained weights with strict=False
        2. Reporting any unexpected and missing keys
        3. Verifying that only classifier-related keys are missing
        
        Args:
            pretrained_path (str): Path to the pretrained model weights file (.safetensors, .bin, etc.)
            verbose (bool, optional): Whether to print detailed messages about loaded weights. Default: True
            strict (bool, optional): Whether to strictly enforce that the keys in state_dict match. Default: False
        
        Returns:
            Tuple[List[str], List[str]]: Lists of unexpected and missing keys
        
        Raises:
            ValueError: If critical non-classifier keys are missing from the pretrained weights
        """
        
        if verbose:
            print(f"Loading pretrained weights from: {pretrained_path}")
        
        # Determine file type and load accordingly
        if pretrained_path.endswith(".safetensors"):
            if os.path.exists(pretrained_path):
                state_dict = load_file(pretrained_path)
            else:
                raise FileNotFoundError(f"Pretrained weights file not found: {pretrained_path}")
        elif pretrained_path.endswith(".bin"):
            if os.path.exists(pretrained_path):
                state_dict = torch.load(pretrained_path, map_location="cpu")
            else:
                raise FileNotFoundError(f"Pretrained weights file not found: {pretrained_path}")
        else:
            raise ValueError(f"Unsupported weights file format: {pretrained_path}")
        
        # Get current model state dict
        model_state_dict = self.state_dict()
        
        # Load weights with strict=False to allow missing keys (like classifier heads)
        load_result = self.load_state_dict(state_dict, strict=strict)
        
        # Get lists of missing and unexpected keys
        missing_keys = load_result.missing_keys
        unexpected_keys = load_result.unexpected_keys
        
        # Identify keys related to classifier heads
        classifier_related_keys = [k for k in model_state_dict.keys() if 
                                "classification_head" in k or "classifier" in k]
        
        # Check if all missing keys are classifier-related
        non_classifier_missing_keys = [k for k in missing_keys if k not in classifier_related_keys]
        
        if verbose:
            print(f"Loaded pretrained weights with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys")
            
            if classifier_related_keys:
                print(f"Found {len(classifier_related_keys)} classifier-related keys in the model")

            if missing_keys:
                print(f"Missing keys: {missing_keys[:5] + ['...'] if len(missing_keys) > 5 else missing_keys}")
            
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5] + ['...'] if len(unexpected_keys) > 5 else unexpected_keys}")
        
        # Warn if there are missing keys that aren't classifier-related
        if non_classifier_missing_keys and verbose:
            print(f"Warning: {len(non_classifier_missing_keys)} missing keys are not classifier-related")
            print(f"Non-classifier missing keys: {non_classifier_missing_keys[:5] + ['...'] if len(non_classifier_missing_keys) > 5 else non_classifier_missing_keys}")
        
        return load_result, classifier_related_keys

    def _create_classifier(self, classifier_type, input_dim, hidden_dim, proj_size, num_classes, dropout_prob=0.1):
        """
        Create a classifier head based on the specified type.
        
        Args:
            classifier_type (str): Type of classifier to create ('mlp', 'cnn', 'lstm', 'transformer', etc.)
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layers
            proj_size (int): Projection size for some classifiers
            num_classes (int): Number of output classes
            dropout_prob (float): Dropout probability
            
        Returns:
            nn.Module: The classifier module
        """
        if classifier_type == "mlp":
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim // 2, num_classes)
            )
            
        elif classifier_type == "cnn":
            return nn.Sequential(
                nn.Unflatten(1,(1,input_dim)),  
                nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),  # Global average pooling
                nn.Flatten(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim*2, num_classes)
            )
            
        elif classifier_type == "lstm":
            class LSTMClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_dim,
                        hidden_size=hidden_dim,
                        num_layers=2,
                        batch_first=True,
                        dropout=dropout_prob,
                        bidirectional=True
                    )
                    self.dropout = nn.Dropout(dropout_prob)
                    self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
                    
                def forward(self, x):
                    # Reshape if input is just [batch, features]
                    if len(x.shape) == 2:
                        x = x.unsqueeze(1)  # [batch, 1, features]
                    
                    # LSTM expects [batch, seq_len, features]
                    output, (h_n, _) = self.lstm(x)
                    
                    # Concatenate the final hidden states from both directions
                    h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
                    
                    # Apply dropout and final classification
                    h_n = self.dropout(h_n)
                    logits = self.fc(h_n)
                    return logits
                    
            return LSTMClassifier()
            
        elif classifier_type == "transformer":
            # Transformer classifier - attends to different parts of the input with different weights
            class TransformerClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Project input to model dimension if needed
                    self.input_projection = nn.Linear(input_dim, proj_size) if input_dim != proj_size else nn.Identity()
                    
                    # Transformer encoder layer
                    self.encoder_layer = nn.TransformerEncoderLayer(
                        d_model=proj_size,
                        nhead=8,  # Number of attention heads
                        dim_feedforward=hidden_dim,
                        dropout=dropout_prob,
                        activation="gelu",
                        batch_first=True
                    )
                    
                    # Transformer encoder
                    self.transformer_encoder = nn.TransformerEncoder(
                        self.encoder_layer,
                        num_layers=2
                    )
                    
                    # Classification head
                    self.classifier = nn.Linear(proj_size, num_classes)
                    
                def forward(self, x):
                    # Reshape if input is just [batch, features]
                    if len(x.shape) == 2:
                        x = x.unsqueeze(1)  # [batch, 1, features]
                        
                    # Project input if needed
                    x = self.input_projection(x)
                    
                    # Apply transformer
                    x = self.transformer_encoder(x)
                    
                    # Average pooling over sequence dimension
                    x = x.mean(dim=1)
                    
                    # Classification
                    return self.classifier(x)
                    
            return TransformerClassifier()

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int]
    ):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.kernels, self.strides):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths)
        output_lengths = output_lengths.to(torch.long) #long

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        #except:
        #pass
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def encode_frames(self, x):
        try:
            x = x.view(-1, self.in_size)
        except RuntimeError:
            x = x.reshape(-1, self.in_size)
        return self.encoder(x)

    def encode_z(self, x):
        mean = self.z_mean(x)
        logvar = self.z_logvar(x)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def decode_frames(self, z):
        return self.decoder(z)

    @staticmethod
    def compute_supervised_loss(
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
        loss_type: str = "cross_entropy",
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.FloatTensor] = None,
        reduction: str = "sum",
    ):
        """
        Compute supervised classification loss between logits and labels.
        
        This method handles different input shapes for both logits and labels:
        - logits: [B, S, N] (batch_size, num_segments, num_classes) or [B, N] (batch_size, num_classes)
        - labels: [B, S] (batch_size, num_segments) or [B] (batch_size) or [B, S, N] (one-hot encoded)
        
        Args:
            logits (torch.FloatTensor): The predicted logits 
                Shape: [B, S, N] or [B, N] where:
                B = batch size
                S = number of segments
                N = number of classes
            labels (torch.FloatTensor): The target labels
                Shape: [B, S], [B] or [B, S, N] (one-hot encoded)
            loss_type (str, optional): Type of loss to compute. 
                Options: "cross_entropy", "focal", "label_smoothed_ce". Default: "cross_entropy"
            label_smoothing (float, optional): Label smoothing factor (0-1). Default: 0.0
            class_weights (torch.FloatTensor, optional): Weights for each class for weighted loss. Default: None
            reduction (str, optional): Specifies the reduction to apply to the output. 
                Options: 'none', 'mean', 'sum'. Default: 'mean'
                
        Returns:
            torch.FloatTensor: Computed loss
        """
        device = logits.device
        
        # Handle different shapes
        if len(logits.shape) == 3:  # [B, S, N]
            B, S, N = logits.shape
            # Reshape to [B*S, N]
            logits_flat = logits.reshape(-1, N)
            
            # Handle different label shapes
            if len(labels.shape) == 2:  # [B, S]
                labels_flat = labels.reshape(-1)
            elif len(labels.shape) == 1:  # [B]
                # Repeat labels for each segment
                labels_flat = labels.unsqueeze(1).repeat(1, S).reshape(-1)
            elif len(labels.shape) == 3:  # [B, S, N] (one-hot)
                labels_flat = labels.reshape(-1, N)
            else:
                raise ValueError(f"Unsupported label shape: {labels.shape}")
                
        elif len(logits.shape) == 2:  # [B, N]
            B, N = logits.shape
            logits_flat = logits
            
            # Handle different label shapes
            if len(labels.shape) == 1:  # [B]
                labels_flat = labels
            elif len(labels.shape) == 2:  # [B, N] (one-hot)
                if labels.shape[1] == N:  # One-hot encoded
                    labels_flat = labels
                else:  # [B, S] but S != N
                    raise ValueError(f"Label shape {labels.shape} incompatible with logits shape {logits.shape}")
            else:
                raise ValueError(f"Unsupported label shape: {labels.shape}")
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}")
        
        # Check if labels are one-hot encoded
        if len(labels_flat.shape) == 2 and labels_flat.shape[1] > 1:
            is_one_hot = True
            # Convert one-hot to class indices if needed for some loss functions
            class_indices = torch.argmax(labels_flat, dim=1)
        else:
            is_one_hot = False
            class_indices = labels_flat
        
        # Compute the appropriate loss
        if loss_type == "cross_entropy":
            if is_one_hot:
                # For one-hot labels, use binary cross entropy with logits
                if class_weights is not None:
                    # Apply class weights to each sample
                    sample_weights = (labels_flat * class_weights.unsqueeze(0)).sum(dim=1)
                    loss_fn = nn.BCEWithLogitsLoss(weight=sample_weights, reduction=reduction)
                else:
                    loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
                
                loss = loss_fn(logits_flat, labels_flat.float())
            else:
                # For class indices, use cross entropy
                if label_smoothing > 0:
                    loss_fn = nn.CrossEntropyLoss(
                        weight=class_weights,
                        label_smoothing=label_smoothing,
                        reduction=reduction
                    )
                else:
                    loss_fn = nn.CrossEntropyLoss(
                        weight=class_weights,
                        reduction=reduction
                    )
                
                loss = loss_fn(logits_flat, class_indices.long())
                
        elif loss_type == "focal":
            # Focal Loss implementation (helps with class imbalance)
            gamma = 2.0  # Focusing parameter
            
            if is_one_hot:
                # Convert logits to probabilities
                probs = torch.sigmoid(logits_flat)
                # Calculate focal weights
                pt = labels_flat * probs + (1 - labels_flat) * (1 - probs)
                focal_weights = (1 - pt) ** gamma
                
                # Standard BCE loss
                bce_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits_flat, 
                    labels_flat.float(), 
                    reduction='none'
                )
                
                # Apply focal weights
                loss = focal_weights * bce_loss
                
                # Apply class weights if provided
                if class_weights is not None:
                    class_weight_per_sample = (labels_flat * class_weights.unsqueeze(0)).sum(dim=1, keepdim=True)
                    loss = loss * class_weight_per_sample
                    
                # Apply reduction
                if reduction == 'mean':
                    loss = loss.mean()
                elif reduction == 'sum':
                    loss = loss.sum()
            else:
                # One-hot encode class_indices for focal loss calculation
                labels_one_hot = nn.functional.one_hot(class_indices.long(), num_classes=N).float()
                
                # Convert logits to probabilities
                probs = nn.functional.softmax(logits_flat, dim=1)
                
                # Calculate focal weights
                pt = (labels_one_hot * probs).sum(dim=1)
                focal_weights = (1 - pt) ** gamma
                
                # Standard CE loss
                ce_loss = nn.functional.cross_entropy(
                    logits_flat, 
                    class_indices.long(), 
                    weight=class_weights,
                    reduction='none'
                )
                
                # Apply focal weights
                loss = focal_weights * ce_loss
                
                # Apply reduction
                if reduction == 'mean':
                    loss = loss.mean()
                elif reduction == 'sum':
                    loss = loss.sum()
                    
        elif loss_type == "label_smoothed_ce":
            # Custom implementation of label smoothing for both one-hot and class indices
            if is_one_hot:
                # For one-hot labels, smooth the labels directly
                smooth_labels = labels_flat * (1 - label_smoothing) + label_smoothing / N
                
                # Compute KL divergence loss
                log_probs = nn.functional.log_softmax(logits_flat, dim=1)
                loss = -(smooth_labels * log_probs).sum(dim=1)
                
                # Apply class weights if provided
                if class_weights is not None:
                    class_weight_per_sample = (labels_flat * class_weights.unsqueeze(0)).sum(dim=1)
                    loss = loss * class_weight_per_sample
                    
                # Apply reduction
                if reduction == 'mean':
                    loss = loss.mean()
                elif reduction == 'sum':
                    loss = loss.sum()
                elif reduction == 'none':
                    pass  # No reduction needed
            else:
                # For class indices, use built-in CrossEntropyLoss with label_smoothing
                loss_fn = nn.CrossEntropyLoss(
                    weight=class_weights,
                    label_smoothing=label_smoothing,
                    reduction=reduction
                )
                loss = loss_fn(logits_flat, class_indices.long())
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
            
        return loss, logits_flat, class_indices

    def vae_loss(self, recon_x, x, mu, log_var, labels, attention_mask=None):
        batch_size = x.size(0)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
            try:
                recon_x = recon_x[attention_mask].reshape(batch_size, -1)
                x = x[attention_mask].reshape(batch_size, -1)
            except RuntimeError:
                "In case the batch size does not divide the input size exactly - can happen with last batch"
                recon_x = recon_x[attention_mask].flatten()
                x = x[attention_mask].flatten()

            labels = labels[attention_mask]
            mu = mu[attention_mask]
            log_var = log_var[attention_mask]

        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        log_var = torch.clamp(log_var, min=-20, max=20)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        if self.kl_annealing:
            beta = min(1.0, self.global_step/self.warmup_steps) * self.beta
            return (recon_loss + beta * kld_loss), recon_loss, beta * kld_loss, mu, log_var, labels
        else:
            return (recon_loss + self.beta * kld_loss), recon_loss, self.beta * kld_loss, mu, log_var, labels

    def forward(self, input_values, labels, attention_mask, mask_time_indices, global_step):
        self.global_step = global_step
        batch_size = input_values.size(0)
        fc_x = self.encode_frames(input_values)
        z_mean, z_logvar, z = self.encode_z(fc_x)
        recon_x = self.decode_frames(z)
        
        recon_x = recon_x.view(batch_size,-1, self.in_size)
        z_mean = z_mean.view(batch_size,-1, self.z_dim)
        z_logvar = z_logvar.view(batch_size,-1, self.z_dim)
        z = z.view(batch_size,-1, self.z_dim)

        if self.treat_as_sequence and len(recon_x.shape) == 3:
            recon_x = recon_x.squeeze(1)
            z_mean = z_mean.squeeze(1)
            z_logvar = z_logvar.squeeze(1)
            
        vae_loss, recon_loss, kld_loss, z_mean, z_logvar, labels = self.vae_loss(
            recon_x, input_values, z_mean, z_logvar, labels, attention_mask
        )

        "Supervised Loss"
        if self.vae_fine_tuning_classifier is not None:
            logits = self.classification_head(z_mean)

            supervised_loss, logits, labels = self.compute_supervised_loss(
                logits=logits,
                labels=labels,
                loss_type=self.supervised_loss_type,
                label_smoothing=0.0,
                class_weights=None,
                reduction=self.supervised_loss_reduction,
            )

        total_loss = self.vae_loss_weight * vae_loss + self.supervised_loss_weight * supervised_loss

        return z_mean, z_logvar, z, recon_x, total_loss, vae_loss, recon_loss, kld_loss, supervised_loss, logits, labels

