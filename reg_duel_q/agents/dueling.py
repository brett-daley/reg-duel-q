from collections import deque
import pfrl
from pfrl.action_value import ActionValue
from pfrl.agents.dqn import _mean_or_nan
from pfrl.agents import DQN
from pfrl.replay_buffer import batch_experiences
import torch
from typing import Any, Dict, Optional, Tuple, Sequence, List

from reg_duel_q import better_drl_utils

class DuelingDQN(DQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.replay_buffer.num_steps == 1
        self.q_loss_record: deque = deque(maxlen=100)
        self.v_loss_record: deque = deque(maxlen=100)

    def update(
        self, experiences: List[List[Dict[str, Any]]], errors_out: Optional[list] = None
    ) -> None:
        exp_batch = batch_experiences(
            experiences,
            device=self.device,
            phi=self.phi,
            gamma=self.gamma,
            batch_states=self.batch_states,
        )
        v_loss, q_loss = self._compute_loss(exp_batch, errors_out=errors_out)
        loss = v_loss + q_loss

        self.v_loss_record.append(v_loss.item())
        self.q_loss_record.append(q_loss.item())
        self.loss_record.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            pfrl.utils.clip_l2_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optim_t += 1
    
    def _compute_loss(
        self, exp_batch: Dict[str, Any], errors_out: Optional[list] = None
    ) -> torch.Tensor:
        assert not isinstance(self.model, better_drl_utils.ModifiedDuelingNetwork)
        qout = self.model.forward_q(exp_batch["state"], subtract_mean_adv=True)
        q = qout.evaluate_actions(exp_batch["action"])

        self.q_record.extend(q.detach().cpu().numpy().ravel())

        assert errors_out is None
        assert 'weights' not in exp_batch

        q_target = self._compute_target_values(exp_batch)
        q_loss = better_drl_utils.semi_gradient_mse(q_target, q, self.batch_accumulator)
        v_loss = torch.zeros_like(q_loss)
        return v_loss, q_loss

    def _compute_target_values(self, exp_batch: Dict[str, Any]) -> torch.Tensor:
        next_qout = self.target_model.forward_q(exp_batch["next_state"], subtract_mean_adv=True)
        batch_rewards = exp_batch["reward"]
        batch_terminal = exp_batch["is_state_terminal"]
        discount = exp_batch["discount"]
        return batch_rewards + discount * (1.0 - batch_terminal) * next_qout.max

    def _compute_y_and_t(self, exp_batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError  # Not used

    def _evaluate_model_and_update_recurrent_states(self, batch_obs: Sequence[Any]) -> ActionValue:
        batch_xs = self.batch_states(batch_obs, self.device, self.phi)
        assert not self.recurrent
        batch_av = self.model.forward_q(batch_xs)
        return batch_av
    
    def get_statistics(self):
        return super().get_statistics() + [
            ("average_v_loss", _mean_or_nan(self.v_loss_record)),
            ("average_q_loss", _mean_or_nan(self.q_loss_record)),
        ]


class RegularizedDuelingQLearning(DuelingDQN):
    def _compute_loss(
        self, exp_batch: Dict[str, Any], errors_out: Optional[list] = None
    ) -> torch.Tensor:
        assert isinstance(self.model, better_drl_utils.ModifiedDuelingNetwork)
        qout, vout, advout = self.model.forward(exp_batch["state"], return_adv=True)
        q = qout.evaluate_actions(exp_batch["action"])

        self.q_record.extend(q.detach().cpu().numpy().ravel())

        assert errors_out is None
        assert 'weights' not in exp_batch

        q_target = self._compute_target_values(exp_batch)
        l2_penalty = self._l2_penalty(qout, vout, advout)
        q_loss = better_drl_utils.semi_gradient_mse(q_target, q, self.batch_accumulator)
        v_loss = torch.zeros_like(q_loss)
        return v_loss, q_loss + l2_penalty

    def _compute_target_values(self, exp_batch: Dict[str, Any]) -> torch.Tensor:
        next_qout = self.target_model.forward_q(exp_batch["next_state"])
        batch_rewards = exp_batch["reward"]
        batch_terminal = exp_batch["is_state_terminal"]
        discount = exp_batch["discount"]
        return batch_rewards + discount * (1.0 - batch_terminal) * next_qout.max

    def _l2_penalty(self, qout, vout, advout):
        v = vout.q_values.squeeze(dim=-1)
        assert self.batch_accumulator == 'mean'
        sum_squared_adv = torch.sum(torch.square(advout.q_values), dim=-1)
        return 1e-3 * torch.mean(0.5 * (torch.square(v) + sum_squared_adv))
