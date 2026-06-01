"""DLTPKT model for the Statics main experiment."""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import load_problem_skill_mapping


class DLTPKT(nn.Module):
    """Dynamic Learning Transfer Perception Knowledge Tracing."""

    def __init__(
        self,
        data_dir,
        device: torch.device,
        embed_dim: int = 64,
        num_concepts: int = 10,
        gcn_layers: int = 1,
    ) -> None:
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.gcn_layers = gcn_layers
        self.num_concepts = num_concepts
        self.embedding = load_problem_skill_mapping(data_dir, device)
        self.num_questions, self.num_skills = self.embedding.shape

        self.ques_embed = nn.Parameter(torch.rand(self.num_questions, embed_dim))
        nn.init.xavier_uniform_(self.ques_embed)
        self.ans_embed = nn.Embedding(2, embed_dim)
        nn.init.xavier_uniform_(self.ans_embed.weight)
        self.skill_linear = nn.Linear(self.num_skills, embed_dim)
        self.qs_fusion = nn.Linear(2 * embed_dim, embed_dim)

        self.concept_emb = nn.Parameter(torch.rand(num_concepts, embed_dim))
        nn.init.xavier_uniform_(self.concept_emb)
        self.init_concept_state = nn.Parameter(torch.rand(num_concepts, embed_dim))
        nn.init.xavier_uniform_(self.init_concept_state)
        self.concept_update_linear = nn.Linear(2 * embed_dim, embed_dim)
        self.concept_gate_linear = nn.Linear(2 * embed_dim, embed_dim)
        self.concept_fusion = nn.Linear(2 * embed_dim, embed_dim)
        self.concept_broadcast_gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

        self.init_transfer_state = nn.Parameter(torch.zeros(self.num_skills, embed_dim))
        nn.init.xavier_uniform_(self.init_transfer_state)
        self.init_concept_transfer_state = nn.Parameter(torch.zeros(num_concepts, embed_dim))
        nn.init.xavier_uniform_(self.init_concept_transfer_state)
        self.transfer_update_linear = nn.Linear(4 * embed_dim, embed_dim)
        self.concept_transfer_update_linear = nn.Linear(4 * embed_dim, embed_dim)
        self.transfer_update_gate = nn.Linear(4 * embed_dim, 1)
        self.concept_transfer_update_gate = nn.Linear(4 * embed_dim, 1)
        self.transfer_adj_query = nn.Linear(embed_dim, embed_dim)
        self.transfer_adj_key = nn.Linear(embed_dim, embed_dim)
        self.concept_transfer_adj_query = nn.Linear(embed_dim, embed_dim)
        self.concept_transfer_adj_key = nn.Linear(embed_dim, embed_dim)

        self.pro_diff_embed = nn.Linear(embed_dim, 1)
        self.skill_mastery_readout = nn.Linear(embed_dim, 1)
        self.concept_mastery_readout = nn.Linear(embed_dim, 1)
        nn.init.xavier_uniform_(self.skill_mastery_readout.weight)
        nn.init.xavier_uniform_(self.concept_mastery_readout.weight)
        self.concept_mastery_influence = nn.Parameter(torch.tensor(1.0))
        self.mastery_ability_layer = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.mastery_ability_layer.weight, 0.5)
        self.residual_ability_layer = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        self.skill_emb = nn.Parameter(torch.rand(self.num_skills, embed_dim))
        nn.init.xavier_uniform_(self.skill_emb)
        self.init_stu_state = nn.Parameter(torch.rand(self.num_skills, embed_dim))
        nn.init.xavier_uniform_(self.init_stu_state)

        # Project shared node embeddings into directed structural roles.
        self.skill_pre_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.skill_succ_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.concept_pre_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.concept_succ_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.skill_pre_projection.weight)
        nn.init.xavier_uniform_(self.skill_succ_projection.weight)
        nn.init.xavier_uniform_(self.concept_pre_projection.weight)
        nn.init.xavier_uniform_(self.concept_succ_projection.weight)

        self.update_linear = nn.Linear(2 * embed_dim, embed_dim)
        self.gate_linear = nn.Linear(2 * embed_dim, embed_dim)
        self.time_embed_layer = nn.Linear(1, embed_dim)
        nn.init.xavier_uniform_(self.time_embed_layer.weight)
        self.forget_gate_skill = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.forget_gate_skill.weight)
        self.forget_gate_concept = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.forget_gate_concept.weight)

    def forward(
        self,
        questions: torch.Tensor,
        answers: torch.Tensor,
        next_questions: torch.Tensor,
        timestamps: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        question_embed = F.embedding(questions, self.ques_embed)
        answer_embed = self.ans_embed(answers)
        next_question_embed = F.embedding(next_questions, self.ques_embed)

        skill_embed = self.skill_linear(F.embedding(questions, self.embedding))
        next_skill_embed = self.skill_linear(F.embedding(next_questions, self.embedding))
        question_skill_embed = self._fuse_question_skill(question_embed, skill_embed)
        next_question_skill_embed = self._fuse_question_skill(next_question_embed, next_skill_embed)
        interactions = torch.cat((question_skill_embed, answer_embed), dim=-1)

        batch_size, seq_len, _ = interactions.size()
        dt_seq = self._normalized_time_deltas(timestamps)
        time_embed = self.time_embed_layer(dt_seq.unsqueeze(-1))
        skill_forget_gate = torch.exp(-F.softplus(self.forget_gate_skill(time_embed)) * dt_seq.unsqueeze(-1))
        concept_forget_gate = torch.exp(-F.softplus(self.forget_gate_concept(time_embed)) * dt_seq.unsqueeze(-1))

        skill_pre_roles = self.skill_pre_projection(self.skill_emb)
        skill_succ_roles = self.skill_succ_projection(self.skill_emb)
        concept_pre_roles = self.concept_pre_projection(self.concept_emb)
        concept_succ_roles = self.concept_succ_projection(self.concept_emb)
        skill_adj = self._static_adjacency(skill_succ_roles, skill_pre_roles)
        concept_adj = self._static_adjacency(concept_succ_roles, concept_pre_roles)
        skill_concept_attention = torch.softmax(
            torch.matmul(self.skill_emb, self.concept_emb.t()) / (self.embed_dim ** 0.5),
            dim=-1,
        )

        skill_state = self.init_stu_state.unsqueeze(0).expand(batch_size, -1, -1).clone()
        concept_state = self.init_concept_state.unsqueeze(0).expand(batch_size, -1, -1).clone()
        transfer_state = self.init_transfer_state.unsqueeze(0).expand(batch_size, -1, -1).clone()
        concept_transfer_state = (
            self.init_concept_transfer_state.unsqueeze(0).expand(batch_size, -1, -1).clone()
        )
        difficulty_seq = self.pro_diff_embed(next_question_skill_embed)
        logits_seq = []
        skill_mastery_seq = []
        concept_mastery_seq = []
        target_skill_mastery_seq = []
        target_concept_mastery_seq = []
        mastery_ability_seq = []
        residual_ability_seq = []

        for step in range(seq_len):
            interaction = interactions[:, step, :]
            skill_weights = F.embedding(questions[:, step], self.embedding)
            skill_weights_norm = skill_weights / (skill_weights.sum(1, keepdim=True) + 1e-9)
            concept_weights = torch.matmul(skill_weights_norm, skill_concept_attention)
            concept_weights_norm = concept_weights / (concept_weights.sum(1, keepdim=True) + 1e-9)

            skill_state = skill_state * skill_forget_gate[:, step].unsqueeze(1)
            concept_state = concept_state * concept_forget_gate[:, step].unsqueeze(1)
            skill_state_before_adaptation = skill_state.clone()
            concept_state_before_adaptation = concept_state.clone()

            concept_state = self._adapt_state(
                concept_state,
                concept_weights.unsqueeze(-1),
                interaction,
                self.concept_update_linear,
                self.concept_gate_linear,
            )
            skill_state = self._adapt_state(
                skill_state,
                skill_weights.unsqueeze(-1),
                interaction,
                self.update_linear,
                self.gate_linear,
            )

            skill_context_before = self._weighted_state(skill_state_before_adaptation, skill_weights_norm)
            skill_context_after = self._weighted_state(skill_state, skill_weights_norm)
            concept_context_before = self._weighted_state(concept_state_before_adaptation, concept_weights_norm)
            concept_context_after = self._weighted_state(concept_state, concept_weights_norm)
            transfer_input = torch.cat((interaction, skill_context_before, skill_context_after), dim=-1)
            concept_transfer_input = torch.cat(
                (interaction, concept_context_before, concept_context_after),
                dim=-1,
            )

            transfer_state = self._update_transfer_state(
                transfer_state,
                skill_weights.unsqueeze(-1),
                transfer_input,
                self.transfer_update_linear,
                self.transfer_update_gate,
            )
            concept_transfer_state = self._update_transfer_state(
                concept_transfer_state,
                concept_weights.unsqueeze(-1),
                concept_transfer_input,
                self.concept_transfer_update_linear,
                self.concept_transfer_update_gate,
            )

            transfer_capability = self._dynamic_transfer_adjacency(
                transfer_state,
                self.transfer_adj_query,
                self.transfer_adj_key,
            )
            concept_transfer_capability = self._dynamic_transfer_adjacency(
                concept_transfer_state,
                self.concept_transfer_adj_query,
                self.concept_transfer_adj_key,
            )
            skill_state = self._propagate(skill_state, skill_adj.unsqueeze(0) * transfer_capability)
            concept_state = self._propagate(
                concept_state,
                concept_adj.unsqueeze(0) * concept_transfer_capability,
            )

            concept_broadcast = torch.matmul(skill_concept_attention, concept_state)
            broadcast_gate = self.concept_broadcast_gate(torch.cat((skill_state, concept_broadcast), dim=-1))
            skill_state = skill_state + broadcast_gate * (concept_broadcast - skill_state)

            next_skill_weights = F.embedding(next_questions[:, step], self.embedding)
            state_read_skill = self._weighted_state(skill_state, next_skill_weights)
            next_skill_weights_norm = next_skill_weights / (next_skill_weights.sum(1, keepdim=True) + 1e-9)
            next_concept_weights = torch.matmul(next_skill_weights_norm, skill_concept_attention)
            state_read_concept = self._weighted_state(concept_state, next_concept_weights)

            concept_mastery = torch.sigmoid(self.concept_mastery_readout(concept_state)).squeeze(-1)
            concept_support = torch.matmul(
                skill_concept_attention,
                concept_mastery.unsqueeze(-1),
            ).squeeze(-1)
            skill_mastery = torch.sigmoid(
                self.skill_mastery_readout(skill_state).squeeze(-1)
                + self.concept_mastery_influence * concept_support
            )
            target_skill_mastery = self._weighted_scalar(skill_mastery, next_skill_weights)
            target_concept_mastery = self._weighted_scalar(concept_mastery, next_concept_weights)

            mastery_ability = self.mastery_ability_layer(
                torch.stack((target_skill_mastery, target_concept_mastery), dim=-1)
            )
            state_read = self.concept_fusion(torch.cat((state_read_skill, state_read_concept), dim=-1))
            residual_ability = self.residual_ability_layer(
                torch.cat((state_read, next_question_skill_embed[:, step, :]), dim=-1)
            )
            logits_seq.append(
                5.0 * (
                    mastery_ability
                    + residual_ability
                    - difficulty_seq[:, step, :]
                )
            )

            if return_details:
                skill_mastery_seq.append(skill_mastery)
                concept_mastery_seq.append(concept_mastery)
                target_skill_mastery_seq.append(target_skill_mastery)
                target_concept_mastery_seq.append(target_concept_mastery)
                mastery_ability_seq.append(mastery_ability.squeeze(-1))
                residual_ability_seq.append(residual_ability.squeeze(-1))

        predictions = torch.sigmoid(torch.stack(logits_seq, dim=1)).squeeze(-1)
        if not return_details:
            return predictions

        details = {
            "skill_mastery": torch.stack(skill_mastery_seq, dim=1),
            "concept_mastery": torch.stack(concept_mastery_seq, dim=1),
            "target_skill_mastery": torch.stack(target_skill_mastery_seq, dim=1),
            "target_concept_mastery": torch.stack(target_concept_mastery_seq, dim=1),
            "mastery_ability": torch.stack(mastery_ability_seq, dim=1),
            "residual_ability": torch.stack(residual_ability_seq, dim=1),
            "difficulty": difficulty_seq.squeeze(-1),
        }
        return predictions, details

    def _fuse_question_skill(self, question_embed: torch.Tensor, skill_embed: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.qs_fusion(torch.cat((question_embed, skill_embed), dim=-1)))
        return gate * question_embed + (1.0 - gate) * skill_embed

    @staticmethod
    def _normalized_time_deltas(timestamps: torch.Tensor) -> torch.Tensor:
        timestamps = timestamps.squeeze(-1)
        dt_seq = torch.relu(timestamps[:, 1:] - timestamps[:, :-1])
        dt_mean = dt_seq.mean(dim=1, keepdim=True).clamp_min(1e-6)
        normalized = dt_seq / dt_mean
        return torch.cat([torch.zeros(timestamps.size(0), 1, device=timestamps.device), normalized], dim=1)

    @staticmethod
    def _static_adjacency(successor_embed: torch.Tensor, predecessor_embed: torch.Tensor) -> torch.Tensor:
        logits = torch.matmul(successor_embed, predecessor_embed.t())
        diagonal = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(diagonal, -1e9)
        soft_adjacency = torch.sigmoid(logits * 5.0)
        hard_adjacency = (soft_adjacency > 0.6).float()

        # Keep a sparse binary graph in the forward pass while allowing the
        # shared embedding projections to receive gradients during training.
        adjacency = hard_adjacency.detach() - soft_adjacency.detach() + soft_adjacency
        return adjacency / (adjacency.sum(dim=-1, keepdim=True) + 1e-9)

    @staticmethod
    def _weighted_state(state: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (state * weights.unsqueeze(-1)).sum(1) / (weights.sum(1, keepdim=True) + 1e-9)

    @staticmethod
    def _weighted_scalar(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (values * weights).sum(1) / (weights.sum(1) + 1e-9)

    @staticmethod
    def _adapt_state(
        state: torch.Tensor,
        mask: torch.Tensor,
        interaction: torch.Tensor,
        update_layer: nn.Module,
        gate_layer: nn.Module,
    ) -> torch.Tensor:
        update = torch.tanh(update_layer(interaction)).unsqueeze(1)
        gate = torch.sigmoid(gate_layer(interaction)).unsqueeze(1)
        adapted = (1.0 - gate) * state + gate * update
        return (1.0 - mask) * state + mask * adapted

    @staticmethod
    def _update_transfer_state(
        state: torch.Tensor,
        mask: torch.Tensor,
        transfer_input: torch.Tensor,
        update_layer: nn.Module,
        gate_layer: nn.Module,
    ) -> torch.Tensor:
        update = torch.tanh(update_layer(transfer_input)).unsqueeze(1)
        gate = torch.sigmoid(gate_layer(transfer_input)).unsqueeze(1)
        adapted = (1.0 - gate) * state + gate * update
        return (1.0 - mask) * state + mask * adapted

    @staticmethod
    def _dynamic_transfer_adjacency(
        state: torch.Tensor,
        query_layer: nn.Module,
        key_layer: nn.Module,
    ) -> torch.Tensor:
        query = F.normalize(query_layer(state), p=2, dim=-1)
        key = F.normalize(key_layer(state), p=2, dim=-1)
        return torch.sigmoid(torch.matmul(query, key.transpose(-1, -2)) * 10.0)

    def _propagate(self, state: torch.Tensor, effective_adj: torch.Tensor) -> torch.Tensor:
        inflow = state
        for _ in range(self.gcn_layers):
            inflow = torch.matmul(effective_adj, inflow)
        outflow_factor = effective_adj.sum(dim=-1, keepdim=True)
        return state + outflow_factor * (inflow - state)
