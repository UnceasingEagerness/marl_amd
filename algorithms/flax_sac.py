import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple

class DeepSetOAB(nn.Module):
    """LiDAR processor for JAX. (Original Basic MLP)"""
    num_points: int = 64
    out_features: int = 64

    @nn.compact
    def __call__(self, x):
        # x is flat LiDAR array [B, num_points] containing normalized ranges
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_features)(x)
        x = nn.relu(x)
        return x

class SigmoidOAB(nn.Module):
    """IEEE Paper Implementation: Sigmoid Gating Mechanism for LiDAR."""
    num_points: int = 64
    out_features: int = 64

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(64)(x)
        gate = nn.sigmoid(h)
        feat = nn.relu(h)
        gated_feat = gate * feat
        out = nn.Dense(self.out_features)(gated_feat)
        return nn.relu(out)

class CNNOAB(nn.Module):
    """IEEE Paper Implementation: 2-Frame CNN LiDAR Encoder."""
    out_features: int = 64

    @nn.compact
    def __call__(self, x_seq):
        # x_seq shape expected: [B, 2, 64] (2 frames, 64 horizontal LiDAR beams)
        # Transpose to channel-last format for Conv1D: [B, 64, 2]
        x = jnp.transpose(x_seq, (0, 2, 1))
        
        # Conv Layer 1: 32 channels, kernel 5, stride 2, circular padding
        # Pad width = (kernel_size - 1) // 2 = 2
        x_pad = jnp.pad(x, ((0,0), (2,2), (0,0)), mode='wrap')
        h1 = nn.Conv(features=32, kernel_size=(5,), strides=(2,), padding='VALID')(x_pad)
        h1 = nn.relu(h1)
        
        # Conv Layer 2: 64 channels, kernel 3, stride 2, circular padding
        # Pad width = (kernel_size - 1) // 2 = 1
        x_pad2 = jnp.pad(h1, ((0,0), (1,1), (0,0)), mode='wrap')
        h2 = nn.Conv(features=64, kernel_size=(3,), strides=(2,), padding='VALID')(x_pad2)
        h2 = nn.relu(h2)
        
        # Flatten and map to output features
        flat = h2.reshape((x.shape[0], -1))
        out = nn.Dense(self.out_features)(flat)
        return nn.relu(out)

class EntitySetEncoder(nn.Module):
    """Deep Sets Encoder for permutation-invariant entity sets."""
    embed_dim: int = 64

    @nn.compact
    def __call__(self, entities, query_features=None):
        # entities shape: [B, num_entities, feature_dim]
        # features: [active_flag, rx, ry, rvx, rvy]
        mask = entities[:, :, 0] > 0.5
        features = entities[:, :, 1:]
        
        # phi network
        h = nn.Dense(self.embed_dim)(features)
        h = nn.relu(h)
        h = nn.Dense(self.embed_dim)(h)
        h = nn.relu(h)
        
        # mask out inactive entities
        h = h * jnp.expand_dims(mask, -1)
        
        # mean pool
        active_counts = jnp.clip(jnp.sum(mask, axis=1, keepdims=True), 1.0, None)
        pooled = jnp.sum(h, axis=1) / active_counts
        
        # rho network
        out = nn.Dense(self.embed_dim)(pooled)
        out = nn.relu(out)
        out = nn.Dense(self.embed_dim)(out)
        
        # If no entities exist, zero out the feature
        has_entities = jnp.expand_dims(jnp.any(mask, axis=1), -1)
        return jnp.where(has_entities, out, jnp.zeros_like(out))

class GNNAttentionEncoder(nn.Module):
    """Graph Neural Network Encoder using Multi-Head Cross-Attention."""
    embed_dim: int = 64
    num_heads: int = 4

    @nn.compact
    def __call__(self, entities, query_features):
        # entities shape: [B, num_entities, feature_dim]
        # query_features shape: [B, query_dim]
        mask = entities[:, :, 0] > 0.5
        features = entities[:, :, 1:]
        
        # Embed the neighbor features (Keys and Values)
        kv = nn.Dense(self.embed_dim)(features)
        kv = nn.relu(kv)
        kv = nn.Dense(self.embed_dim)(kv)
        
        # Embed the query (Ego Kinematics)
        q = nn.Dense(self.embed_dim)(query_features)
        # Add sequence dimension for attention [B, 1, embed_dim]
        q = jnp.expand_dims(q, axis=1)
        
        # Cross-Attention mask: True where valid. shape: [B, 1, 1, num_entities]
        attn_mask = jnp.expand_dims(mask, axis=(1, 2))
        
        # MultiHeadDotProductAttention computes weighted sum of neighbors based on relevance to Ego
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, 
            qkv_features=self.embed_dim, 
            out_features=self.embed_dim
        )(inputs_q=q, inputs_kv=kv, mask=attn_mask)
        
        # Squeeze the sequence dimension back out -> [B, embed_dim]
        attn_out = jnp.squeeze(attn_out, axis=1)
        
        # Final non-linear processing
        out = nn.Dense(self.embed_dim)(attn_out)
        out = nn.relu(out)
        out = nn.Dense(self.embed_dim)(out)
        
        # If no entities exist, zero out the feature
        has_entities = jnp.expand_dims(jnp.any(mask, axis=1), -1)
        return jnp.where(has_entities, out, jnp.zeros_like(out))

class ActorBackbone(nn.Module):
    """[LEGACY (Mean Pool)] Decentralized Actor Feature Extractor."""
    layout: dict

    @nn.compact
    def __call__(self, x):
        ego_spec = self.layout["ego"]
        goal_spec = self.layout["goal"]
        lidar_spec = self.layout["lidar"]
        auv_spec = self.layout["auv_entities"]
        mob_spec = self.layout["moving_obstacles"]
        
        def slice_vector(name):
            start = self.layout[name]["start"]
            dim = self.layout[name]["dim"]
            return x[:, start:start+dim]
            
        def slice_entities(name):
            start = self.layout[name]["start"]
            dim = self.layout[name]["dim"]
            count = self.layout[name]["count"]
            feat_dim = self.layout[name]["feature_dim"]
            flat = x[:, start:start+dim]
            return flat.reshape((x.shape[0], count, feat_dim))

        kin_feat = nn.Dense(64)(slice_vector("ego"))
        kin_feat = nn.LayerNorm()(kin_feat)
        kin_feat = nn.relu(kin_feat)
        kin_feat = nn.Dense(64)(kin_feat)
        kin_feat = nn.relu(kin_feat)
        
        goal_feat = nn.Dense(32)(slice_vector("goal"))
        goal_feat = nn.LayerNorm()(goal_feat)
        goal_feat = nn.relu(goal_feat)
        
        auv_feat = EntitySetEncoder(embed_dim=64)(slice_entities("auv_entities"))
        moving_feat = EntitySetEncoder(embed_dim=64)(slice_entities("moving_obstacles"))
        
        lidar_feat = DeepSetOAB(num_points=lidar_spec["dim"]//2, out_features=64)(slice_vector("lidar"))
        
        fused = nn.LayerNorm()(jnp.concatenate([kin_feat, lidar_feat], axis=1))
        combined = jnp.concatenate([fused, goal_feat, auv_feat, moving_feat], axis=1)
        
        out = nn.Dense(256)(combined)
        out = nn.relu(out)
        return out

# =========================================================================================
# VARIANT 2: SPATIO-TEMPORAL LSTM ARCHITECTURE (70m Max Pool + Frame Stacking)
# =========================================================================================

class EntityMaxEncoder(nn.Module):
    """Deep Sets Encoder using Max Pooling and a strict 70m LiDAR Range Filter."""
    embed_dim: int = 64

    @nn.compact
    def __call__(self, entities, query_features=None):
        # entities shape: [B, num_entities, feature_dim]
        # features: [active_flag, rx, ry, rvx, rvy]
        mask = entities[:, :, 0] > 0.5
        features = entities[:, :, 1:]
        
        # 1. The 70m Observability Filter
        rx = features[:, :, 0]
        ry = features[:, :, 1]
        dist = jnp.sqrt(rx**2 + ry**2)
        in_range_mask = dist <= 70.0
        
        final_mask = jnp.logical_and(mask, in_range_mask)
        
        # phi network
        h = nn.Dense(self.embed_dim)(features)
        h = nn.relu(h)
        h = nn.Dense(self.embed_dim)(h)
        h = nn.relu(h)
        
        # Mask out inactive OR out-of-range entities.
        # By setting them to a large negative number, MaxPool will naturally ignore them.
        h = jnp.where(jnp.expand_dims(final_mask, -1), h, -1e9)
        
        # Max Pool (replaces the flawed Mean Pool)
        # We must handle the edge case where the entity array is strictly size 0 (e.g. moving_obstacles)
        if h.shape[1] == 0:
            pooled = jnp.zeros((h.shape[0], h.shape[2]))
        else:
            pooled = jnp.max(h, axis=1)
        
        # rho network
        out = nn.Dense(self.embed_dim)(pooled)
        out = nn.relu(out)
        out = nn.Dense(self.embed_dim)(out)
        
        # If no entities were in range, zero out the feature
        has_entities = jnp.expand_dims(jnp.any(final_mask, axis=1), -1)
        return jnp.where(has_entities, out, jnp.zeros_like(out))

class TemporalActorBackbone(nn.Module):
    """Spatio-Temporal Actor using Frame Stacking and an LSTM."""
    layout: dict
    seq_len: int = 10

    @nn.compact
    def __call__(self, x_flat):
        B = x_flat.shape[0]
        x_seq = x_flat.reshape((B, self.seq_len, 92))
        
        class SpatialExtractor(nn.Module):
            layout: dict
            @nn.compact
            def __call__(self, x):
                def slice_vector(name):
                    start = self.layout[name]["start"]
                    dim = self.layout[name]["dim"]
                    return x[:, start:start+dim]
                    
                def slice_entities(name):
                    start = self.layout[name]["start"]
                    dim = self.layout[name]["dim"]
                    count = self.layout[name]["count"]
                    feat_dim = self.layout[name]["feature_dim"]
                    flat = x[:, start:start+dim]
                    return flat.reshape((x.shape[0], count, feat_dim))

                kin_feat = nn.Dense(64)(slice_vector("ego"))
                kin_feat = nn.LayerNorm()(kin_feat)
                kin_feat = nn.relu(kin_feat)
                kin_feat = nn.Dense(64)(kin_feat)
                kin_feat = nn.relu(kin_feat)
                
                goal_feat = nn.Dense(32)(slice_vector("goal"))
                goal_feat = nn.LayerNorm()(goal_feat)
                goal_feat = nn.relu(goal_feat)
                
                # Use our NEW EntityMaxEncoder for strict spatial awareness
                auv_feat = EntityMaxEncoder(embed_dim=64)(slice_entities("auv_entities"))
                moving_feat = EntityMaxEncoder(embed_dim=64)(slice_entities("moving_obstacles"))
                
                lidar_feat = DeepSetOAB(num_points=self.layout["lidar"]["dim"]//2, out_features=64)(slice_vector("lidar"))
                
                fused = nn.LayerNorm()(jnp.concatenate([kin_feat, lidar_feat], axis=1))
                combined = jnp.concatenate([fused, goal_feat, auv_feat, moving_feat], axis=1)
                
                out = nn.Dense(128)(combined)
                out = nn.relu(out)
                return out

        # Vectorize across the seq_len dimension using vmap
        VmappedSpatial = nn.vmap(
            SpatialExtractor,
            variable_axes={'params': None}, 
            split_rngs={'params': False},
            in_axes=1,  
            out_axes=1
        )
        
        spatial_seq = VmappedSpatial(layout=self.layout)(x_seq) # [B, seq_len, 128]
        
        # Temporal Memory (LSTM)
        LSTM = nn.RNN(nn.OptimizedLSTMCell(features=128), return_carry=True)
        (lstm_carry, lstm_hidden), lstm_out = LSTM(spatial_seq)
        
        out = nn.Dense(256)(lstm_hidden)
        out = nn.relu(out)
        return out

class SoftQNetwork(nn.Module):
    layout: dict
    
    @nn.compact
    def __call__(self, x, a):
        # We use the STAE Actor Backbone for the critic as well
        features = STAE_ActorBackbone(layout=self.layout)(x)
        x = jnp.concatenate([features, a], axis=1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

class Actor(nn.Module):
    layout: dict
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    
    @nn.compact
    def __call__(self, x):
        features = STAE_Max_ActorBackbone(layout=self.layout)(x)
        mean = nn.Dense(self.action_dim)(features)
        log_std = nn.Dense(self.action_dim)(features)
        log_std = jnp.clip(log_std, -5.0, 2.0)
        return mean, log_std

    def get_action(self, x, key):
        mean, log_std = self(x)
        std = jnp.exp(log_std)
        normal = mean + std * jax.random.normal(key, mean.shape)
        action = jnp.tanh(normal)
        action_env = action * self.action_scale + self.action_bias
        # We need the log_prob
        log_prob = jax.scipy.stats.norm.logpdf(normal, mean, std) - jnp.log(1 - action**2 + 1e-6)
        log_prob = jnp.sum(log_prob, axis=1, keepdims=True)
        return action_env, log_prob

# =========================================================================================
# VARIANT 3: SPATIO-TEMPORAL ATTENTION ENCODER (STAE)
# =========================================================================================

class FlaxSpatioTemporalAttentionEncoder(nn.Module):
    """Flax Spatio-Temporal Attention Encoder based on the Reference Diagram."""
    embed_dim: int = 64
    hidden_dim: int = 128
    
    @nn.compact
    def __call__(self, ego_seq, entity_seqs):
        # ego_seq: [B, seq_len, 8]  (Kinematics)
        # entity_seqs: [B, seq_len, num_entities, 5] (Traffic/Neighbors)
        B, seq_len, num_entities, _ = entity_seqs.shape
        
        # 1. Ego Temporal Encoding (Query)
        ego_emb = nn.Dense(self.embed_dim)(ego_seq)
        ego_emb = nn.relu(ego_emb)
        LSTM_ego = nn.RNN(nn.OptimizedLSTMCell(self.hidden_dim), return_carry=True)
        (ego_carry, ego_hidden), _ = LSTM_ego(ego_emb)
        ego_q = jnp.expand_dims(ego_hidden, axis=1) # [B, 1, hidden_dim] -> Query Q
        
        # 2. Entity Temporal Encoding (Keys/Values)
        entity_flat = entity_seqs.reshape((B * num_entities, seq_len, -1))
        # Mask out inactive entities, but keep their features for now
        entity_feat = entity_flat[:, :, 1:] # [B*num, seq_len, 4]
        
        entity_emb = nn.Dense(self.embed_dim)(entity_feat)
        entity_emb = nn.relu(entity_emb)
        LSTM_shared = nn.RNN(nn.OptimizedLSTMCell(self.hidden_dim), return_carry=True)
        (ent_carry, ent_hidden), _ = LSTM_shared(entity_emb)
        ent_kv = ent_hidden.reshape((B, num_entities, self.hidden_dim)) # [B, num_entities, hidden_dim] -> Keys K / Values V
        
        # Attention Mask (True if active/valid). entity_seqs[..., 0] is the active flag.
        key_mask = entity_seqs[:, -1, :, 0] > 0.5 # [B, num_entities]
        
        # Handle case where num_entities is 0
        if num_entities == 0:
            attn_output = jnp.zeros_like(ego_q)
        else:
            # MultiHeadDotProductAttention requires mask shape: [batch, num_heads, q_seq_len, kv_seq_len]
            # We expand to [B, 1, 1, num_entities] to broadcast across heads and q_seq_len (which is 1)
            key_mask = jnp.expand_dims(key_mask, axis=(1, 2))
            # 3. Cross-Attention
            attn_output = nn.MultiHeadDotProductAttention(num_heads=4, qkv_features=self.hidden_dim)(
                inputs_q=ego_q, inputs_kv=ent_kv, mask=key_mask
            )
        
        # 4. Add & Norm (Residual)
        ego_interactive_enc = nn.LayerNorm()(jnp.squeeze(ego_q, 1) + jnp.squeeze(attn_output, 1)) # [B, hidden_dim]
        
        # 5. FC Layer
        z_t = nn.Dense(128)(ego_interactive_enc)
        z_t = nn.relu(z_t)
        
        return z_t

class STAE_ActorBackbone(nn.Module):
    """Integrates STAE with Context Encoders (LiDAR & Goal)."""
    layout: dict
    seq_len: int = 10

    @nn.compact
    def __call__(self, x_flat):
        B = x_flat.shape[0]
        x_seq = x_flat.reshape((B, self.seq_len, 92))
        
        def slice_vector_seq(name):
            start = self.layout[name]["start"]
            dim = self.layout[name]["dim"]
            return x_seq[:, :, start:start+dim]
            
        def slice_entities_seq(name):
            start = self.layout[name]["start"]
            dim = self.layout[name]["dim"]
            count = self.layout[name]["count"]
            feat_dim = self.layout[name]["feature_dim"]
            flat = x_seq[:, :, start:start+dim]
            return flat.reshape((B, self.seq_len, count, feat_dim))
            
        # TOP BRANCH: STAE (Spatio-Temporal Attention Encoder)
        ego_seq = slice_vector_seq("ego") # [B, seq_len, 8]
        auv_seqs = slice_entities_seq("auv_entities") # [B, seq_len, 4, 5]
        moving_seqs = slice_entities_seq("moving_obstacles") # [B, seq_len, 0, 5]
        
        # Combine all entities into one traffic array
        traffic_seqs = jnp.concatenate([auv_seqs, moving_seqs], axis=2)
        
        z_t = FlaxSpatioTemporalAttentionEncoder(embed_dim=64, hidden_dim=128)(ego_seq, traffic_seqs) # [B, 128]
        
        # BOTTOM BRANCH: Context Encoders (Using ONLY current frame t=0, which is the last frame in the sequence)
        current_frame = x_seq[:, -1, :]
        
        def slice_current(name):
            start = self.layout[name]["start"]
            dim = self.layout[name]["dim"]
            return current_frame[:, start:start+dim]
            
        # Goal MLP
        goal_feat = nn.Dense(32)(slice_current("goal"))
        goal_feat = nn.LayerNorm()(goal_feat)
        goal_feat = nn.relu(goal_feat)
        
        # LiDAR Dual-Stream OAB -> replaced by CNNOAB (Temporal 2-Frame Stack)
        # We slice the last 2 frames from the LiDAR sequence
        lidar_seq = slice_vector_seq("lidar")[:, -2:, :] # [B, 2, 64]
        lidar_feat = CNNOAB(out_features=64)(lidar_seq)
        
        # Fusion C_t
        c_t = jnp.concatenate([goal_feat, lidar_feat], axis=1) # [B, 96]
        
        # Final Fusion S_t
        s_t = jnp.concatenate([z_t, c_t], axis=1) # [B, 128 + 96] = [B, 224]
        
        out = nn.Dense(256)(s_t)
        out = nn.relu(out)
        return out

# =========================================================================================
# VARIANT 4: STAE + ENTITY MAX ENCODER (Fix for Teammate Collision Blurring)
# =========================================================================================

class STAE_Max_ActorBackbone(nn.Module):
    """VARIANT 4: Integrates STAE with EntityMaxEncoder to prevent attention blurring."""
    layout: dict
    seq_len: int = 10

    @nn.compact
    def __call__(self, x_flat):
        B = x_flat.shape[0]
        x_seq = x_flat.reshape((B, self.seq_len, 92))
        
        def slice_vector_seq(name):
            start = self.layout[name]["start"]
            dim = self.layout[name]["dim"]
            return x_seq[:, :, start:start+dim]
            
        def slice_entities_seq(name):
            start = self.layout[name]["start"]
            dim = self.layout[name]["dim"]
            count = self.layout[name]["count"]
            feat_dim = self.layout[name]["feature_dim"]
            flat = x_seq[:, :, start:start+dim]
            return flat.reshape((B, self.seq_len, count, feat_dim))
            
        # TOP BRANCH: STAE (Spatio-Temporal Attention Encoder)
        ego_seq = slice_vector_seq("ego") # [B, seq_len, 8]
        auv_seqs = slice_entities_seq("auv_entities") # [B, seq_len, 4, 5]
        moving_seqs = slice_entities_seq("moving_obstacles") # [B, seq_len, 0, 5]
        
        # Combine all entities into one traffic array
        traffic_seqs = jnp.concatenate([auv_seqs, moving_seqs], axis=2)
        
        z_t = FlaxSpatioTemporalAttentionEncoder(embed_dim=64, hidden_dim=128)(ego_seq, traffic_seqs) # [B, 128]
        
        # BOTTOM BRANCH: Context Encoders (Using ONLY current frame t=0, which is the last frame in the sequence)
        current_frame = x_seq[:, -1, :]
        
        def slice_current(name):
            start = self.layout[name]["start"]
            dim = self.layout[name]["dim"]
            return current_frame[:, start:start+dim]
            
        def slice_entities_current(name):
            start = self.layout[name]["start"]
            dim = self.layout[name]["dim"]
            count = self.layout[name]["count"]
            feat_dim = self.layout[name]["feature_dim"]
            flat = current_frame[:, start:start+dim]
            return flat.reshape((B, count, feat_dim))
            
        # Goal MLP
        goal_feat = nn.Dense(32)(slice_current("goal"))
        goal_feat = nn.LayerNorm()(goal_feat)
        goal_feat = nn.relu(goal_feat)
        
        # LiDAR Dual-Stream OAB -> replaced by CNNOAB (Temporal 2-Frame Stack)
        # We slice the last 2 frames from the LiDAR sequence
        lidar_seq = slice_vector_seq("lidar")[:, -2:, :] # [B, 2, 64]
        lidar_feat = CNNOAB(out_features=64)(lidar_seq)
        
        # EXACT GEOMETRY FIX: EntityMaxEncoder
        auv_feat_max = EntityMaxEncoder(embed_dim=64)(slice_entities_current("auv_entities"))
        
        # Fusion C_t
        c_t = jnp.concatenate([goal_feat, lidar_feat, auv_feat_max], axis=1) # [B, 32 + 64 + 64] = 160
        
        # Final Fusion S_t
        s_t = jnp.concatenate([z_t, c_t], axis=1) # [B, 128 + 160] = 288
        
        out = nn.Dense(256)(s_t)
        out = nn.relu(out)
        return out

# =========================================================================================
# CHAPTER 7 ARCHITECTURE: SWARM TRANSFORMER + GRU
# -----------------------------------------------------------------------------------------
# NOTE: This architecture relies on Self-Attention (all entities attend to all entities)
# and processes temporal sequences using a Gated Recurrent Unit (GRU).
# The input 'x' must have shape [B, seq_len, feature_dim].
# =========================================================================================
#
# class TransformerBlock(nn.Module):
#     embed_dim: int = 64
#     num_heads: int = 4
#     
#     @nn.compact
#     def __call__(self, x, mask=None):
#         # x shape: [B, num_tokens, embed_dim]
#         # mask shape: [B, 1, num_tokens]
#         
#         # 1. Multi-Head Self Attention
#         attn_out = nn.MultiHeadDotProductAttention(
#             num_heads=self.num_heads, 
#             qkv_features=self.embed_dim, 
#             out_features=self.embed_dim
#         )(inputs_q=x, inputs_kv=x, mask=mask)
#         
#         # 2. Residual Add + LayerNorm
#         x = nn.LayerNorm()(x + attn_out)
#         
#         # 3. Feed Forward Network (FFN)
#         ffn_out = nn.Dense(self.embed_dim * 4)(x)
#         ffn_out = nn.relu(ffn_out)
#         ffn_out = nn.Dense(self.embed_dim)(ffn_out)
#         
#         # 4. Residual Add + LayerNorm
#         out = nn.LayerNorm()(x + ffn_out)
#         return out
#
# class SwarmTransformer(nn.Module):
#     embed_dim: int = 64
#     num_heads: int = 4
#     num_layers: int = 2
#     
#     @nn.compact
#     def __call__(self, ego_features, entity_features, entity_mask):
#         # Concatenate Ego token as the first token in the sequence
#         # tokens shape: [B, 1 + num_entities, embed_dim]
#         ego_emb = nn.Dense(self.embed_dim)(ego_features)
#         ego_emb = jnp.expand_dims(ego_emb, axis=1)
#         tokens = jnp.concatenate([ego_emb, entity_features], axis=1)
#         
#         # Create Self-Attention Mask (Ego is always True)
#         ego_mask = jnp.ones((entity_mask.shape[0], 1), dtype=bool)
#         full_mask = jnp.concatenate([ego_mask, entity_mask], axis=1)
#         attn_mask = jnp.expand_dims(full_mask, axis=1)
#         
#         x = tokens
#         for _ in range(self.num_layers):
#             x = TransformerBlock(embed_dim=self.embed_dim, num_heads=self.num_heads)(x, mask=attn_mask)
#             
#         # The updated Ego token contains the global spatial awareness of the entire swarm
#         ego_out = x[:, 0, :]
#         return ego_out
#
# class TransformerGRUActor(nn.Module):
#     embed_dim: int = 64
#     
#     @nn.compact
#     def __call__(self, ego_seq, entity_seqs):
#         # ego_seq: [B, seq_len, ego_dim]
#         # entity_seqs: [B, seq_len, num_entities, feature_dim]
#         B, seq_len, num_entities, _ = entity_seqs.shape
#         
#         # Vectorize the SwarmTransformer across the time (seq_len) dimension using nn.vmap
#         VmappedTransformer = nn.vmap(
#             SwarmTransformer,
#             variable_axes={'params': None}, # Share weights across all time steps
#             split_rngs={'params': False},
#             in_axes=(1, 1, 1),
#             out_axes=1
#         )
#         
#         ent_mask = entity_seqs[:, :, :, 0] > 0.5
#         ent_feat = entity_seqs[:, :, :, 1:]
#         ent_emb = nn.Dense(self.embed_dim)(ent_feat)
#         ent_emb = nn.relu(ent_emb)
#         
#         # spatial_features shape: [B, seq_len, embed_dim]
#         spatial_features = VmappedTransformer(embed_dim=self.embed_dim)(ego_seq, ent_emb, ent_mask)
#         
#         # Process the spatial sequence through a GRU (Gated Recurrent Unit)
#         GRU = nn.RNN(nn.GRUCell(features=128), return_carry=True)
#         (gru_carry, gru_hidden), gru_out = GRU(spatial_features)
#         
#         # The final hidden state contains the complete Spatio-Temporal awareness
#         out = nn.Dense(256)(gru_hidden)
#         out = nn.relu(out)
#         
#         return out

# =========================================================================================
# VARIANT 5: STAE + FiLM GOAL ENCODER + GATED FUSION
# -----------------------------------------------------------------------------------------
# New mechanisms on top of Variant 4:
#   A) Formation-Conditioned Goal Encoder (FiLM):
#      The Goal MLP output is modulated by a compact DeepSet summary of neighbor
#      formation, making the goal embedding aware of swarm geometry.
#   B) Gated Fusion Head:
#      Projects Z_A(128), Z_L(64), Z_G(32) to a common d=64, then uses
#      3 learned softmax scalar gates to produce Z_mission in R^64.
#
# Design invariants:
#   - FiLM projection is zero-initialized -> at step 0, gamma~1, beta~0, Z_G~h (old goal).
#   - N=0 neighbor fallback: s=0 -> FiLM identity, no crash.
#   - EntityMaxEncoder for Z_max kept as parallel safety branch (collision precision).
#   - All existing variants (1-4) are UNTOUCHED for ablation.
# =========================================================================================


class FormationDeepSet(nn.Module):
    """
    Permutation-invariant summary of neighbor formation using mean-pooled MLP.

    Input:  neighbors [B, N_neighbors, 5]  (active_flag, rx, ry, rvx, rvy)
    Output: s in R^embed_dim  (formation context vector, default 16)

    Uses MEAN pool (not sum) so output scale is stable when N varies 5->10.
    If no neighbors are active, returns zero vector -> FiLM reduces to identity.
    """
    embed_dim: int = 16

    @nn.compact
    def __call__(self, neighbors):
        # neighbors: [B, N, 5]
        mask     = neighbors[:, :, 0] > 0.5        # [B, N]  active flag
        features = neighbors[:, :, 1:]             # [B, N, 4]  (rx, ry, rvx, rvy)

        # phi network - shared weights across neighbors
        h = nn.Dense(self.embed_dim)(features)     # [B, N, embed_dim]
        h = nn.relu(h)

        # Zero out inactive neighbors before pooling
        h = h * jnp.expand_dims(mask, -1)

        # Mean pool (safe: divide by max(count, 1))
        active_counts = jnp.clip(jnp.sum(mask, axis=1, keepdims=True), 1.0, None)
        pooled = jnp.sum(h, axis=1) / active_counts   # [B, embed_dim]

        # If no neighbors visible at all, return zeros (FiLM becomes identity)
        has_neighbors = jnp.expand_dims(jnp.any(mask, axis=1), -1)
        pooled = jnp.where(has_neighbors, pooled, jnp.zeros_like(pooled))

        return pooled   # s in R^embed_dim


class FiLMGoalEncoder(nn.Module):
    """
    Formation-conditioned Goal Encoder using Feature-wise Linear Modulation (FiLM).

    Given:
        g  = raw goal vector [B, 3]   (dist_norm, sin_theta, cos_theta)
        s  = formation context [B, 16] (from FormationDeepSet)

    Produces:
        Z_G = gamma * h + beta  in R^goal_dim   (formation-aware goal embedding)
        gamma, beta             returned for diagnostic logging

    The FiLM Dense layer is zero-initialized so at step 0:
        gamma_raw ~ 0  ->  gamma = 1 + tanh(0) = 1
        beta      ~ 0
    meaning Z_G ~ h (identical to Variant 4 goal branch at initialization).
    Training deviates only as the formation signal proves useful.

    gamma in (0, 2) via `1 + tanh(gamma_raw)` -- cannot zero-out or explode goal.
    """
    goal_dim: int = 32

    @nn.compact
    def __call__(self, g, s):
        # Base goal path (unchanged from Variant 4)
        h = nn.Dense(self.goal_dim)(g)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)                              # h in R^32

        # FiLM projection (zero-initialized weights AND bias)
        film_out = nn.Dense(
            self.goal_dim * 2,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(s)                                         # [B, 64]

        gamma_raw = film_out[:, :self.goal_dim]     # [B, 32]
        beta      = film_out[:, self.goal_dim:]     # [B, 32]

        # gamma in (0, 2) -- bounded scale, never zeroes or explodes goal signal
        gamma = 1.0 + jnp.tanh(gamma_raw)          # [B, 32]

        Z_G = gamma * h + beta                      # [B, 32]

        return Z_G, gamma, beta   # return gamma, beta for diagnostics


class GatedFusionHead(nn.Module):
    """
    Projects Z_A, Z_L, Z_G to a common d=64, then fuses with 3 learned
    softmax scalar gates conditioned on the full concatenated context.

    Input dims:  Z_A in R^128,  Z_L in R^64,  Z_G in R^32
    Output:      Z_mission in R^64,  gates in R^3 (for diagnostic logging)

    Scalar (per-branch) gating is easy to log and debug.
    Upgrade path: replace Dense(3) with Dense(192) reshaped to (3, d) and
    softmax over axis=0 per-dimension for per-feature gating.
    """
    d: int = 64

    @nn.compact
    def __call__(self, Z_A, Z_L, Z_G):
        # Project all branches to common dimension
        p_A = nn.Dense(self.d)(Z_A)   # [B, 64]
        p_L = nn.Dense(self.d)(Z_L)   # [B, 64]
        p_G = nn.Dense(self.d)(Z_G)   # [B, 64]

        # Gate computation conditioned on full context
        context     = jnp.concatenate([p_A, p_L, p_G], axis=1)  # [B, 192]
        gate_logits = nn.Dense(3)(context)                        # [B, 3]
        gates       = nn.softmax(gate_logits, axis=-1)            # [B, 3], sums to 1

        g_A = gates[:, 0:1]   # [B, 1]
        g_L = gates[:, 1:2]   # [B, 1]
        g_G = gates[:, 2:3]   # [B, 1]

        Z_mission = g_A * p_A + g_L * p_L + g_G * p_G   # [B, 64]

        return Z_mission, gates   # return gates for diagnostic logging


class STAE_FiLM_ActorBackbone(nn.Module):
    """
    VARIANT 5: Full forward pass combining:
      - FlaxSpatioTemporalAttentionEncoder   -> Z_A in R^128  (temporal dynamics)
      - CNNOAB (2-frame LiDAR CNN)           -> Z_L in R^64   (spatial memory)
      - FiLMGoalEncoder                      -> Z_G in R^32   (formation-aware goal)
      - GatedFusionHead                      -> Z_mission in R^64
      - EntityMaxEncoder                     -> Z_max in R^64  (collision safety aux)

    Final output: concat(Z_mission, Z_max) -> Dense(256) -> ReLU -> R^256
    """
    layout: dict
    seq_len: int = 10

    @nn.compact
    def __call__(self, x_flat):
        B = x_flat.shape[0]
        x_seq = x_flat.reshape((B, self.seq_len, 92))

        # Slice helpers
        def slice_vector_seq(name):
            start = self.layout[name]["start"]
            dim   = self.layout[name]["dim"]
            return x_seq[:, :, start:start+dim]

        def slice_entities_seq(name):
            start    = self.layout[name]["start"]
            dim      = self.layout[name]["dim"]
            count    = self.layout[name]["count"]
            feat_dim = self.layout[name]["feature_dim"]
            flat     = x_seq[:, :, start:start+dim]
            return flat.reshape((B, self.seq_len, count, feat_dim))

        def slice_current(name):
            start = self.layout[name]["start"]
            dim   = self.layout[name]["dim"]
            return x_seq[:, -1, start:start+dim]

        def slice_entities_current(name):
            start    = self.layout[name]["start"]
            dim      = self.layout[name]["dim"]
            count    = self.layout[name]["count"]
            feat_dim = self.layout[name]["feature_dim"]
            flat     = x_seq[:, -1, start:start+dim]
            return flat.reshape((B, count, feat_dim))

        # BRANCH A: STAE -> Z_A in R^128
        ego_seq      = slice_vector_seq("ego")             # [B, 10, 8]
        auv_seqs     = slice_entities_seq("auv_entities")  # [B, 10, N-1, 5]
        moving_seqs  = slice_entities_seq("moving_obstacles")
        traffic_seqs = jnp.concatenate([auv_seqs, moving_seqs], axis=2)

        Z_A = FlaxSpatioTemporalAttentionEncoder(
            embed_dim=64, hidden_dim=128
        )(ego_seq, traffic_seqs)                           # [B, 128]

        # BRANCH B: LiDAR CNN -> Z_L in R^64
        lidar_seq = slice_vector_seq("lidar")[:, -2:, :]   # [B, 2, 64]
        Z_L = CNNOAB(out_features=64)(lidar_seq)           # [B, 64]

        # BRANCH C: Formation-Conditioned Goal -> Z_G in R^32
        g         = slice_current("goal")                          # [B, 3]
        neighbors = slice_entities_current("auv_entities")         # [B, N-1, 5]

        s               = FormationDeepSet(embed_dim=16)(neighbors)   # [B, 16]
        Z_G, gamma, beta = FiLMGoalEncoder(goal_dim=32)(g, s)        # [B, 32]

        # GATED FUSION -> Z_mission in R^64
        Z_mission, gates = GatedFusionHead(d=64)(Z_A, Z_L, Z_G)   # [B, 64]

        # SAFETY AUX: EntityMaxEncoder -> Z_max in R^64
        # Kept alongside gated fusion for hard collision-precision signal
        Z_max = EntityMaxEncoder(embed_dim=64)(neighbors)           # [B, 64]

        # FINAL FUSION: concat -> Dense(256) -> ReLU
        s_t = jnp.concatenate([Z_mission, Z_max], axis=1)          # [B, 128]
        out = nn.Dense(256)(s_t)
        out = nn.relu(out)

        return out   # [B, 256]


# Variant 5 Actor and Critic -------------------------------------------------------

class Actor_FiLM(nn.Module):
    """SAC Actor using Variant 5 (FiLM + Gated Fusion) backbone."""
    layout: dict
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, x):
        features = STAE_FiLM_ActorBackbone(layout=self.layout)(x)
        mean    = nn.Dense(self.action_dim)(features)
        log_std = nn.Dense(self.action_dim)(features)
        log_std = jnp.clip(log_std, -5.0, 2.0)
        return mean, log_std

    def get_action(self, x, key):
        mean, log_std = self(x)
        std    = jnp.exp(log_std)
        normal = mean + std * jax.random.normal(key, mean.shape)
        action = jnp.tanh(normal)
        action_env = action * self.action_scale + self.action_bias
        log_prob = (
            jax.scipy.stats.norm.logpdf(normal, mean, std)
            - jnp.log(1 - action**2 + 1e-6)
        )
        log_prob = jnp.sum(log_prob, axis=1, keepdims=True)
        return action_env, log_prob


class SoftQNetwork_FiLM(nn.Module):
    """SAC Critic using Variant 5 (FiLM + Gated Fusion) backbone."""
    layout: dict

    @nn.compact
    def __call__(self, x, a):
        features = STAE_FiLM_ActorBackbone(layout=self.layout)(x)
        x = jnp.concatenate([features, a], axis=1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x
