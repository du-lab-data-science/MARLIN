# Lightning module for the MARLIN block-diffusion decoder.

import contextlib
from typing import Optional

import hydra.utils
import lightning as L
import torch
import torch.nn as nn
from transformers import BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig

from dlm.bert_with_cross_attention import BertForMaskedLMWithCrossAttention
from dlm.utils.ema import ExponentialMovingAverage
from dlm.utils.utils_data import get_tokenizer
from dlm.utils.utils_save import clean_checkpoint, fast_forward_info

from marlin import block_diffusion as bd
from marlin.chem.mass_table import build_mass_table
from marlin.conditioning import MassIsoFpConditioner


class MarlinDecoder(L.LightningModule):
    """Mass + isotope + fingerprint conditioned block-diffusion decoder."""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.tokenizer = get_tokenizer(config.data.get("hf_cache_dir", None))
        self.mask_index = self.tokenizer.mask_token_id
        self.bos_index = self.tokenizer.bos_token_id
        self.eos_index = self.tokenizer.eos_token_id
        self.pad_index = self.tokenizer.pad_token_id

        m = config.model
        self.hidden_size = m.hidden_size
        self.block_size = int(m.get("block_size", 8))

        self.use_mass = m.get("use_mass_conditioning", True)
        self.use_isotope = m.get("use_isotope_conditioning", True)
        self.use_fingerprint = m.get("use_fingerprint_conditioning", True)

        bert_config_dict = dict(m)
        for k in ("lr_scheduler",):
            bert_config_dict.pop(k, None)
        # keep only BERT keys
        for k in list(bert_config_dict.keys()):
            if k.startswith(("mass_", "isotope_", "fingerprint_", "use_", "block_")):
                bert_config_dict.pop(k, None)
        bert_config = BertConfig.from_dict(bert_config_dict)
        bert_config._attn_implementation = "eager"  # custom 2D additive masks

        self.conditioner = MassIsoFpConditioner(
            hidden_size=m.hidden_size,
            num_attention_heads=m.num_attention_heads,
            num_layers=m.num_hidden_layers,
            dropout=m.hidden_dropout_prob,
            use_mass=self.use_mass,
            use_isotope=self.use_isotope,
            use_fingerprint=self.use_fingerprint,
            mass_num_bands=m.get("mass_num_bands", 32),
            mass_min=m.get("mass_min", 0.0),
            mass_max=m.get("mass_max", 2000.0),
            mass_hidden_dim=m.get("mass_hidden_dim", 256),
            isotope_hidden_dim=m.get("isotope_hidden_dim", 64),
            fingerprint_bits=m.get("fingerprint_bits", 4096),
            fingerprint_max_seq_len=m.get("fingerprint_seq_max_len", 256),
            fingerprint_num_self_attention_layers=m.get("fingerprint_num_self_attention_layers", 3),
            fingerprint_activation_threshold=m.get("fingerprint_activation_threshold", 0.0),
            mass_dropout_prob=m.get("mass_dropout_prob", 0.1),
            isotope_dropout_prob=m.get("isotope_dropout_prob", 0.1),
            fingerprint_dropout_prob=m.get("fingerprint_dropout_prob", 0.25),
        )

        any_cond = self.use_mass or self.use_isotope or self.use_fingerprint
        if any_cond:
            self.backbone = BertForMaskedLMWithCrossAttention(
                bert_config,
                cross_attention_layers=self.conditioner.cross_attention_layers,
                fingerprint_cross_attention_layers=None,
                use_shared_cross_attention=True,
            )
        else:  # unconditional
            self.backbone = BertForMaskedLM(bert_config)

        self.register_buffer("m_heavy", build_mass_table(self.tokenizer))

        # before EMA, so EMA tracks only trainable parameters
        self._apply_freezing(config.training.get("freeze_mode", "none"))

        if config.training.get("ema", 0) and config.training.ema > 0:
            self.ema = ExponentialMovingAverage(self.parameters(), decay=config.training.ema)
        else:
            self.ema = None

        self.fast_forward_epochs = None
        self.fast_forward_batches = None

    def _apply_freezing(self, mode: str):
        """Optionally freeze pretrained weights (``none``, ``peft`` or ``heads_only``)."""
        mode = (mode or "none").lower()
        if mode == "none":
            return

        def is_backbone_block(n):
            return n.startswith("backbone.bert.encoder.layer.") and ".bert_layer." in n
        def is_embeddings(n):
            return n.startswith("backbone.bert.embeddings.")
        def is_cross_attn(n):
            return (".cross_attention." in n and n.startswith("backbone.")) or \
                   n.startswith("conditioner.cross_attention_layers.")
        def is_mlm_head(n):
            return n.startswith("backbone.cls.")
        def is_fp_encoder(n):
            return n.startswith("conditioner.fingerprint_encoder.")
        def is_new_head(n):
            return n.startswith("conditioner.mass_conditioner.") or \
                   n.startswith("conditioner.isotope_conditioner.")

        frozen, trainable = 0, 0
        for name, p in self.named_parameters():
            if is_new_head(name):
                keep = True
            elif mode == "heads_only":
                keep = False
            elif mode == "peft":
                keep = is_cross_attn(name) or is_mlm_head(name)
            else:
                raise ValueError(f"unknown freeze_mode: {mode}")
            p.requires_grad_(keep)
            if keep:
                trainable += p.numel()
            else:
                frozen += p.numel()
        print(f"[MarlinDecoder] freeze_mode={mode}: trainable={trainable/1e6:.2f}M  frozen={frozen/1e6:.2f}M")

    def _trainable(self):
        return self.parameters()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # m_heavy is recomputed below, so it may be missing from the checkpoint
        safe_missing = {"m_heavy"}
        model_keys = set(self.state_dict().keys())
        missing = model_keys - set(state_dict.keys())
        use_strict = strict and not (missing and missing <= safe_missing)
        incompatible = super().load_state_dict(state_dict, strict=use_strict, assign=assign)
        self.register_buffer("m_heavy", build_mass_table(self.tokenizer).to(self.device))
        return incompatible

    def _token_embeddings(self, ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        emb_layer = self.backbone.bert.embeddings
        word = emb_layer.word_embeddings(ids)
        if position_ids is None:
            position_ids = torch.arange(ids.size(1), device=ids.device).unsqueeze(0).expand_as(ids)
        pos = emb_layer.position_embeddings(position_ids)
        tok_type = emb_layer.token_type_embeddings(torch.zeros_like(ids))
        return word + pos + tok_type

    def _encode(self, embeddings, additive_mask, condition_embeddings, condition_mask):
        emb_layer = self.backbone.bert.embeddings
        h = emb_layer.LayerNorm(embeddings)
        h = emb_layer.dropout(h)
        kwargs = {}
        if condition_embeddings is not None:
            kwargs["condition_embeddings"] = condition_embeddings
            kwargs["condition_mask"] = condition_mask
        encoder_outputs = self.backbone.bert.encoder(h, attention_mask=additive_mask, **kwargs)
        sequence_output = encoder_outputs[0]
        return self.backbone.cls(sequence_output)

    def _autocast(self):
        if torch.cuda.is_available():
            return torch.amp.autocast("cuda", dtype=torch.float32)
        return contextlib.nullcontext()

    def _conditioning(self, batch_size, device, mass, isotope, fingerprint, training, fingerprint_mask=None):
        return self.conditioner.encode(
            batch_size=batch_size,
            device=device,
            mass=mass,
            isotope=isotope,
            fingerprint=fingerprint,
            fingerprint_mask=fingerprint_mask,
            training=training,
        )

    def denoise_single(self, x, additive_mask, mass=None, isotope=None, fingerprint=None,
                       cond=None):
        """Logits ``[B, L, V]`` for a single stream (inference)."""
        with self._autocast():
            if cond is None:
                cond = self._conditioning(x.size(0), x.device, mass, isotope, fingerprint, self.training)
            Z, Zmask = cond
            emb = self._token_embeddings(x)
            return self._encode(emb, additive_mask, Z, Zmask)

    def denoise_two_stream(self, x0, xt, additive_mask_2l, cond):
        """Two-stream training forward; returns the logits of the noised half ``[B, L, V]``."""
        B, L = x0.shape
        ids = torch.cat([x0, xt], dim=1)
        pos = torch.arange(L, device=x0.device)
        position_ids = torch.cat([pos, pos], dim=0).unsqueeze(0).expand(B, -1)
        with self._autocast():
            Z, Zmask = cond
            emb = self._token_embeddings(ids, position_ids)
            logits = self._encode(emb, additive_mask_2l, Z, Zmask)
        return logits[:, L:, :]

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        B, L = input_ids.shape
        device = input_ids.device

        mass = batch.get("mass", None) if self.use_mass else None
        isotope = batch.get("isotope", None) if self.use_isotope else None
        fingerprint = batch.get("fingerprint", None) if self.use_fingerprint else None
        # 0 skips the fingerprint token
        fingerprint_mask = batch.get("fingerprint_mask", None) if self.use_fingerprint else None

        # held-out (val/test) molecules contribute no loss
        exclude_mask = batch.get("exclude_mask", None)
        if exclude_mask is not None:
            attention_mask = attention_mask * exclude_mask.to(device).unsqueeze(1)

        block_id = bd.make_block_ids(L, self.block_size, device)
        num_blocks = int(block_id.max().item()) + 1

        # BOS (position 0) is always-clean context; pads are never noised.
        noiseable = attention_mask.bool().clone()
        noiseable[:, 0] = False

        cond = self._conditioning(B, device, mass, isotope, fingerprint, training=True,
                                  fingerprint_mask=fingerprint_mask)
        scheme = self.config.training.get("block_scheme", "two_stream")
        antithetic = self.config.training.get("antithetic_sampling", True)
        eps = self.config.training.get("sampling_eps", 1e-3)

        if scheme == "two_stream":
            t_per_block = bd.sample_block_times(B, num_blocks, device, eps=eps, antithetic=antithetic)
            t_pos = bd.times_to_positions(t_per_block, block_id)
            xt, mask_bool = bd.noise_masked(input_ids, t_pos, self.mask_index, noiseable)
            add_mask = bd.build_two_stream_mask(block_id, attention_mask, B, device)
            logits = self.denoise_two_stream(input_ids, xt, add_mask, cond)
        else:  # single_block
            t_scalar = bd.sample_block_times(B, 1, device, eps=eps, antithetic=antithetic).squeeze(1)
            # pick one block per example that contains >=1 noiseable position
            has_noise = torch.stack(
                [(noiseable & (block_id.unsqueeze(0) == k)).any(dim=1) for k in range(num_blocks)],
                dim=1,
            ).float()
            has_noise = has_noise + 1e-6
            chosen = torch.multinomial(has_noise, 1).squeeze(1)
            in_chosen = block_id.unsqueeze(0) == chosen.unsqueeze(1)
            t_pos = t_scalar.unsqueeze(1).expand(B, L)
            block_noiseable = noiseable & in_chosen
            xt, mask_bool = bd.noise_masked(input_ids, t_pos, self.mask_index, block_noiseable)
            add_mask = bd.build_block_causal_mask(block_id, attention_mask, B, device)
            logits = self.denoise_single(xt, add_mask, cond=cond)

        loss, per_block = bd.diffusion_nll(
            logits, input_ids, mask_bool, t_pos, block_id, num_blocks, eps=eps
        )

        self.log("train_loss", loss.item(), on_step=True, prog_bar=True, sync_dist=True)
        for k, v in enumerate(per_block.tolist()):
            self.log(f"loss_block_{k}", v, on_step=True, prog_bar=False, sync_dist=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", current_lr, prog_bar=False, sync_dist=True)
        return loss

    def on_after_backward(self):
        if self.global_step % 20 == 0:
            grads = [p.grad.detach() for p in self.parameters() if p.grad is not None]
            if grads:
                total = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
                self.log("grad_norm", total, on_step=True, sync_dist=True)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )
        if hasattr(self.config.model, "lr_scheduler"):
            sc = self.config.model.lr_scheduler
            warmup_steps = getattr(sc, "warmup_steps", None)
            if warmup_steps is not None:
                from torch.optim.lr_scheduler import LinearLR, SequentialLR

                main = hydra.utils.instantiate(
                    {"_target_": sc._target_, "T_max": sc.T_max, "eta_min": sc.eta_min, "optimizer": optimizer}
                )
                warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
                scheduler = SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_steps])
            else:
                scheduler = hydra.utils.instantiate(sc, optimizer=optimizer)
            return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "lr"}]
        scheduler = hydra.utils.instantiate(
            {"_target_": "transformers.get_constant_schedule_with_warmup", "num_warmup_steps": 2500},
            optimizer=optimizer,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "lr"}]

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(self.parameters())

    def on_train_start(self):
        self.backbone.train()
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)

    def on_load_checkpoint(self, checkpoint):
        if self.ema and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
        self.fast_forward_epochs, self.fast_forward_batches = fast_forward_info(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint["ema"] = self.ema.state_dict()
        clean_checkpoint(checkpoint, self.trainer.accumulate_grad_batches)
