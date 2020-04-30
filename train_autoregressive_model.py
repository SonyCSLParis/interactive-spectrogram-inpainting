from typing import Optional, Iterable
from datetime import datetime
import uuid
import argparse
import os
import pathlib
import json
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from radam import RAdam

try:
    from apex import amp
except ModuleNotFoundError:
    amp = None

from torch.utils.tensorboard.writer import SummaryWriter

from dataset import LMDBDataset
from pixelsnail import PixelSNAIL, LabelSmoothingLoss
from transformer import VQNSynthTransformer
from scheduler import CycleScheduler, get_cosine_schedule_with_warmup
from sequence_mask import (SequenceMask, BernoulliSequenceMask,
                           UniformProbabilityBernoulliSequenceMask,
                           UniformMaskedAmountSequenceMask,
                           ContiguousZonesSequenceMask)

# use matplotlib without an X server
# on desktop, this prevents matplotlib windows from popping around
mpl.use('Agg')

DIRPATH = os.path.dirname(os.path.abspath(__file__))

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


def plot_codes(target_codemaps: torch.LongTensor,
               predicted_codemaps: torch.LongTensor,
               success_maps: torch.FloatTensor,
               codes_dictionary_dim: int,
               cmap='viridis', plots_per_row: int = 12) -> None:
    assert (len(target_codemaps)
            == len(predicted_codemaps)
            == len(success_maps))

    num_maps = len(target_codemaps)
    num_groups = 3
    plots_per_row = min(num_maps, plots_per_row)
    num_rows_per_codemaps_group = int(np.ceil(num_maps / plots_per_row))
    num_rows = num_groups * num_rows_per_codemaps_group

    figure, subplot_axs = plt.subplots(num_rows, plots_per_row,
                                       figsize=(10, 2*num_rows))
    for ax in subplot_axs.ravel().tolist():
        ax.set_axis_off()

    def get_ax(codemap_group_index: int, codemap_index: int):
        start_row = codemap_group_index * num_rows_per_codemaps_group
        row = start_row + codemap_index // plots_per_row
        ax = subplot_axs[row][codemap_index % plots_per_row]
        return ax

    for group_index, maps_group in enumerate([target_codemaps,
                                              predicted_codemaps]):
        for map_index, codemap in enumerate(maps_group):
            ax = get_ax(group_index, map_index)
            im = ax.matshow(codemap.cpu().numpy(), vmin=0,
                            vmax=codes_dictionary_dim-1,
                            cmap=cmap)

    # print success maps
    codemap_group_index = 2
    for map_index, success_map in enumerate(success_maps):
        ax = get_ax(codemap_group_index, map_index)
        ax.matshow(success_map.cpu().numpy(), vmin=0, vmax=1,
                   cmap='RdYlGn')

    figure.tight_layout()
    # add colorbar for codemaps
    figure.colorbar(im,
                    ax=(subplot_axs.ravel().tolist()))
    return figure, subplot_axs


def num_samples_in_loader(loader: torch.utils.data.DataLoader):
    if loader.drop_last:
        return len(loader.dataset)
    else:
        batch_size = loader.batch_size
        return len(loader) * batch_size


def num_satisfied_constraints(predicted: torch.LongTensor,
                              condition: torch.LongTensor,
                              mask: torch.BoolTensor,
                              ):
    """Check that a self-conditional model correctly sets the constraints
    """
    # positions where the predicted map matches the condition
    correct = torch.eq(predicted, condition).float()
    # ignore mistakes where there was no constraint
    satisfied_constraints = torch.masked_fill(correct, mask, 1)
    return satisfied_constraints.sum()


def run_model(args, epoch: int, loader: DataLoader, model: VQNSynthTransformer,
              optimizer, scheduler, device,
              criterion: nn.Module,
              tensorboard_writer: Optional[SummaryWriter] = None,
              is_training: bool = True,
              mask_sampler: Optional[SequenceMask] = None,
              plot_frequency_batch: int = 200,
              num_codes_dictionary: int = None,
              clip_grad_norm: Optional[float] = None,
              drop_loss_half_DEBUG: bool = False,
              train_num_steps_sequences_DEBUG: Optional[int] = None):
    run_type = 'training' if is_training else 'validation'
    status_bar = tqdm(total=0, position=0, bar_format='{desc}')
    tqdm_loader = tqdm(loader, position=1)
    num_training_samples = len(loader.dataset)

    loss_sum = 0
    total_accuracy = 0
    num_samples_seen_epoch = 0
    # number of samples seen across runs, useful for TensorBoard tracking
    num_samples_seen_total = epoch * num_samples_in_loader(loader)

    satisfied_constraints_total = None
    if model.self_conditional_model:
        satisfied_constraints_total = 0

    if is_training:
        model = model.train()
    else:
        model = model.eval()

    parallel_model = nn.DataParallel(model)

    for batch_index, (top, bottom, class_conditioning_tensors) in enumerate(tqdm_loader):
        if is_training:
            parallel_model.zero_grad()

        class_conditioning_tensors = {
            condition_name: condition_tensor.to(device, non_blocking=True)
            for condition_name, condition_tensor
            in class_conditioning_tensors.items()}

        if (model.self_conditional_model
                and model.local_class_conditioning):
            class_conditioning_tensors = {
                key: tensor.view(-1, 1, 1).repeat(
                    1,
                    model.source_frequencies,
                    model.source_duration)
                for key, tensor in class_conditioning_tensors.items()
            }
        else:
            class_conditioning_tensors = {
                key: tensor.view(-1, 1)
                for key, tensor in class_conditioning_tensors.items()
            }

        top = top.to(device, non_blocking=True)

        if args.hier == 'top':
            if model.self_conditional_model:
                kind = 'target'
                source = target = top

                # apply masking to self-conditioning
                batch_size = top.shape[0]
                mask = mask_sampler.sample_mask(batch_size)

                if model.local_class_conditioning:
                    class_condition_sequence = (
                        model.make_condition_sequence(
                            class_conditioning_tensors
                            )
                        )
                else:
                    class_condition_sequence = None

                masked_source_sequence, target_sequence = (
                    model.to_sequences(
                        target, condition=source,
                        class_conditioning=class_conditioning_tensors,
                        mask=mask)
                    )

                logits_sequence_out, _ = parallel_model(
                    target_sequence, condition=masked_source_sequence,
                    class_condition=class_condition_sequence)
            else:
                kind = 'source'
                target = top
                source_sequence, _ = (
                    model.to_sequences(
                        top, condition=None,
                        class_conditioning=class_conditioning_tensors
                        ))

                logits_sequence_out, _ = parallel_model(
                    source_sequence, condition=None)

        elif args.hier == 'bottom':
            kind = 'target'
            bottom = bottom.to(device, non_blocking=True)
            target = bottom
            source_sequence, target_sequence = (
                model.to_sequences(
                    bottom, condition=top,
                    class_conditioning=class_conditioning_tensors)
            )

            logits_sequence_out, _ = parallel_model(
                target_sequence,
                condition=source_sequence)

        time_frequency_logits_out = model.to_time_frequency_map(
            logits_sequence_out, kind=kind, permute_output_as_logits=True)

        if train_num_steps_sequences_DEBUG:
            time_frequency_map_out = time_frequency_logits_out.permute(
                0, 2, 3, 1)
            logits_sequence_out = model.flatten_map(
                time_frequency_map_out, kind=kind).transpose(1, 2)
            target_sequence = model.flatten_map(
                target, kind=kind)

            loss = criterion(
                logits_sequence_out[..., :train_num_steps_sequences_DEBUG],
                target_sequence[..., :train_num_steps_sequences_DEBUG]
                )
        elif drop_loss_half_DEBUG:
            loss = criterion(
                time_frequency_logits_out[..., :model.shape[1]//2],
                target[..., :model.shape[1]//2])
        else:
            loss = criterion(time_frequency_logits_out, target)

        if is_training:
            nn.utils.clip_grad_norm_(parallel_model.parameters(),
                                     clip_grad_norm)

            loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        _, pred = time_frequency_logits_out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / correct.numel()
        if model.self_conditional_model:
            time_frequency_mask = model.to_time_frequency_map(
                mask, kind='source')
            satisfied_constraints_batch = num_satisfied_constraints(
                pred, source, time_frequency_mask)
            satisfied_constraints_total += satisfied_constraints_batch

        lr = optimizer.param_groups[0]['lr']

        batch_size = top.shape[0]
        loss_sum += loss.item() * batch_size
        total_accuracy += accuracy * batch_size
        num_samples_seen_epoch += batch_size
        num_samples_seen_total += batch_size

        status_bar.set_description_str(
            (
                f'{run_type}, epoch: {epoch + 1}; avg loss: {loss_sum / num_samples_seen_epoch:.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.8f}'
            )
        )

        if is_training and tensorboard_writer is not None:
            # report metrics per batch
            loss_name = str(criterion)
            tensorboard_writer.add_scalar(
                f'code_prediction-{run_type}_{args.hier}-{num_training_samples}_training_samples/{loss_name}',
                loss,
                num_samples_seen_total)
            tensorboard_writer.add_scalar(
                f'code_prediction-{run_type}_{args.hier}-{num_training_samples}_training_samples/accuracy',
                accuracy,
                num_samples_seen_total)
            if model.self_conditional_model:
                tensorboard_writer.add_scalar(
                    (f'code_prediction-{run_type}_{args.hier}-{num_training_samples}'
                     '_training_samples/satisfied_constraints_ratio'),
                    satisfied_constraints_batch / top.numel(),
                    num_samples_seen_total)

        if tensorboard_writer is not None and batch_index % plot_frequency_batch == 0:
            num_plot_samples = min(batch_size, 10)

            correct_bool = (target == pred)
            correct_float = correct_bool.float()

            if model.self_conditional_model:
                # represent success map with four values/colors using the mask
                # shades describe the following cases
                # [wrong masked, wrong unmasked, correct unmasked, correct masked]
                unmasked_map = torch.logical_not(time_frequency_mask)
                # correct, masked values
                correct_float.masked_fill_(unmasked_map * correct_bool, 0.8)
                # incorrect, masked values
                correct_float.masked_fill_(
                    unmasked_map * torch.logical_not(correct_bool),
                    0.2)

            # one row of input codemaps and one row of model-output codemaps
            fig_codes, _ = plot_codes(target[:num_plot_samples],
                                      pred[:num_plot_samples],
                                      correct_float[:num_plot_samples],
                                      num_codes_dictionary,
                                      plots_per_row=num_plot_samples)
            if run_type == 'validation':
                plot_suptitle = (f'{run_type.capitalize()}: Target and predicted codes, success map\n'
                                 f"(after {epoch} training epoch{'s' if epoch > 1 else ''} of training)")
                tensorboard_plot_tag = f'code_prediction-{run_type}_{args.hier}/Codes-Target-Output'
            else:
                plot_suptitle = (f'{run_type.capitalize()}: Target and predicted codes, success map\n'
                                 f"(after {num_training_samples} training samples)")
                tensorboard_plot_tag = f'code_prediction-{run_type}_{args.hier}-{num_training_samples}_training_samples/Codes-Target-Output'

            fig_codes.suptitle(plot_suptitle)
            tensorboard_writer.add_figure(
                tensorboard_plot_tag,
                fig_codes,
                num_samples_seen_total
            )

    if not is_training and tensorboard_writer is not None:
        # only report metrics over full validation/test set
        loss_name = str(criterion)
        tensorboard_writer.add_scalar(
            (f'code_prediction-{run_type}_{args.hier}'
             + (('-' + mask_sampler.__class__.__name__) if mask_sampler is not None else '')
             + '/mean_{loss_name}'),
            loss_sum / num_samples_seen_epoch,
            epoch)
        tensorboard_writer.add_scalar(
            (f'code_prediction-{run_type}_{args.hier}'
             + (('-' + mask_sampler.__class__.__name__) if mask_sampler is not None else '')
             + '/mean_accuracy'),
            total_accuracy / num_samples_seen_epoch,
            epoch)
        if model.self_conditional_model:
            tensorboard_writer.add_scalar(
                (f'code_prediction-{run_type}_{args.hier}'
                 + (('-' + mask_sampler.__class__.__name__) if mask_sampler is not None else '')
                 + '/satisfied_constraints_ratio'),
                satisfied_constraints_total / num_samples_seen_epoch,
                epoch)

    return loss_sum, total_accuracy, num_samples_seen_epoch


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=420)
    parser.add_argument('--model_type', type=str,
                        choices=['PixelSNAIL', 'Transformer'],
                        default='PixelSNAIL')
    parser.add_argument('--optimizer', type=str,
                        choices=['adam', 'radam'],
                        default='adam')
    parser.add_argument('--optimizer_eps', type=float, default=1e-8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hier', type=str, default='top')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--clip_grad_norm', type=float, default=None)
    parser.add_argument('--channel', type=int, default=256)
    parser.add_argument('--label_smoothing', default=0.0, type=float)
    parser.add_argument('--n_res_block', type=int, default=4)
    parser.add_argument('--n_res_channel', type=int, default=256)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=3)
    parser.add_argument('--positional_embeddings_dim', type=int, default=16)
    parser.add_argument('--classes_for_conditioning', type=str,
                        nargs='*', default=['instrument_family_str', 'pitch'])
    parser.add_argument('--class_conditioning_embedding_dim_per_modality',
                        type=int, default=16)
    parser.add_argument('--class_conditioning_prepend_to_dummy_input',
                        action='store_true')
    parser.add_argument('--self_conditional_model', action='store_true',
                        help=('whether to use an encoder/decoder architecture'
                              'with masked self-supervision'))
    parser.add_argument('--use_identity_memory_mask', action='store_true')
    parser.add_argument('--use_relative_transformer', action='store_true')
    parser.add_argument('--use_local_class_conditioning', action='store_true')
    parser.add_argument('--positional_class_conditioning', action='store_true')
    parser.add_argument('--conditional_model_nhead', type=int, default=16)
    parser.add_argument('--conditional_model_num_encoder_layers', type=int,
                        default=6)
    parser.add_argument('--conditional_model_num_decoder_layers', type=int,
                        default=8)
    parser.add_argument('--unconditional_model_nhead', type=int, default=8)
    parser.add_argument('--unconditional_model_num_encoder_layers', type=int,
                        default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--predict_frequencies_first', action='store_true')
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--scheduler', type=str)
    parser.add_argument('--num_warmup_steps', type=int, default=2000)
    parser.add_argument('--initial_weights_path', type=str,
                        help=("Restart training from the weights "
                              "contained in the provided PyTorch checkpoint"))
    parser.add_argument('--disable_writes_to_disk', action='store_true')
    parser.add_argument('--disable_tensorboard', action='store_true')
    parser.add_argument('--plot_frequency_batch', type=int, default=200)
    parser.add_argument('--database_path', type=str, required=True)
    parser.add_argument('--validation_database_path', type=str, default=None)
    parser.add_argument('--evaluate_only', action='store_true')
    parser.add_argument('--num_training_samples', type=int,
                        help=('If provided, trims to input dataset to only use'
                              'the given number of samples'))
    parser.add_argument('--vqvae_run_id', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for the Dataloaders')
    parser.add_argument('--mask_sampling_strategy', type=str,
                        choices=['bernoulli', 'random_p_bernoulli',
                                 'uniform_masked_amount', 'contiguous_zones'],
                        default='random_p_bernoulli')
    parser.add_argument('--bernoulli_masking_probability', type=float)
    parser.add_argument('--random_p_bernoulli_p_range', type=float, nargs=2,
                        default=[0, 1])
    parser.add_argument('--uniform_masked_amount_min_masking_ratio',
                        type=float, default=0.)
    parser.add_argument('--disable_start_symbol_DEBUG', action='store_true')
    parser.add_argument(
        '--drop_loss_half_DEBUG', action='store_true',
        help="""If set, ignore the second half (in time) of the codemaps,
        which often contains a lot of silence-mapped symbols and could lead
        the training to fail""")
    parser.add_argument(
        '--train_num_steps_sequences_DEBUG', type=int,
        help="If set, restrict training sequences to their first `n` steps")

    args = parser.parse_args()

    print(args)

    if args.model_type == 'PixelSNAIL':
        prediction_model = PixelSNAIL
    elif args.model_type == 'Transformer':
        prediction_model = VQNSynthTransformer

    run_ID = (f'{args.model_type}-{args.hier}_layer-'
              + datetime.now().strftime('%Y%m%d-%H%M%S-')
              + str(uuid.uuid4())[:6])

    print("Run ID: ", run_ID)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    DATABASE_PATH = pathlib.Path(args.database_path)
    dataset = LMDBDataset(
        DATABASE_PATH.expanduser().absolute(),
        classes_for_conditioning=args.classes_for_conditioning
    )

    num_training_samples = args.num_training_samples
    if num_training_samples is None:
        # use all available training samples
        num_training_samples = len(dataset)

    loader = DataLoader(
        torch.utils.data.Subset(dataset, range(num_training_samples)),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
        pin_memory=True,
    )
    class_conditioning_num_classes_per_modality = {
        modality: len(label_encoder.classes_)
        for modality, label_encoder in dataset.label_encoders.items()
    }

    class_conditioning_embedding_dim_per_modality = {
        modality: args.class_conditioning_embedding_dim_per_modality
        for modality in dataset.label_encoders.keys()
    }

    validation_loader = None
    if args.validation_database_path is not None:
        validation_dataset = LMDBDataset(
            args.validation_database_path,
            classes_for_conditioning=args.classes_for_conditioning)
        validation_loader = DataLoader(
            validation_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=False
        )

    model_checkpoint_weights = None
    if args.initial_weights_path is not None:
        model_checkpoint_weights = torch.load(
            args.initial_weights_path)
        # if 'args' in model_checkpoint_weights:
        #     args = model_checkpoint_weights['args']

    shape_top, shape_bottom = (list(dataset[0][i].shape) for i in range(2))
    if args.hier == 'top':
        model = prediction_model(
            shape=shape_top,
            n_class=512,
            channel=args.channel,
            kernel_size=5,
            n_block=4,
            n_res_block=args.n_res_block,
            res_channel=args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
            positional_embeddings_dim=args.positional_embeddings_dim,

            use_relative_transformer=args.use_relative_transformer,
            predict_frequencies_first=args.predict_frequencies_first,

            conditional_model=args.self_conditional_model,
            self_conditional_model=args.self_conditional_model,
            add_mask_token_to_symbols=args.self_conditional_model,
            condition_shape=shape_top if args.self_conditional_model else None,
            local_class_conditioning=args.use_local_class_conditioning,
            positional_class_conditioning=args.positional_class_conditioning,

            class_conditioning_num_classes_per_modality=(
                class_conditioning_num_classes_per_modality),
            class_conditioning_embedding_dim_per_modality=(
                class_conditioning_embedding_dim_per_modality),
            class_conditioning_prepend_to_dummy_input=(
                args.class_conditioning_prepend_to_dummy_input),

            use_identity_memory_mask=args.use_identity_memory_mask,

            conditional_model_nhead=args.conditional_model_nhead,
            conditional_model_num_encoder_layers=(
                args.conditional_model_num_encoder_layers),
            conditional_model_num_decoder_layers=(
                args.conditional_model_num_decoder_layers),
            unconditional_model_nhead=args.unconditional_model_nhead,
            unconditional_model_num_encoder_layers=(
                args.unconditional_model_num_encoder_layers),
            disable_start_symbol_DEBUG=args.disable_start_symbol_DEBUG,
        )
    elif args.hier == 'bottom':
        model = prediction_model(
            shape=shape_bottom,
            n_class=512,
            channel=args.channel,
            kernel_size=5,
            n_block=4,
            n_res_block=args.n_res_block,
            res_channel=args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
            positional_embeddings_dim=args.positional_embeddings_dim,

            use_relative_transformer=args.use_relative_transformer,
            predict_frequencies_first=args.predict_frequencies_first,

            conditional_model=True,
            self_conditional_model=False,
            condition_shape=shape_top,
            conditional_model_nhead=args.conditional_model_nhead,
            conditional_model_num_encoder_layers=(
                args.conditional_model_num_encoder_layers),
            conditional_model_num_decoder_layers=(
                args.conditional_model_num_decoder_layers),
            use_identity_memory_mask=args.use_identity_memory_mask,
            local_class_conditioning=args.use_local_class_conditioning,
            positional_class_conditioning=args.positional_class_conditioning,

            class_conditioning_num_classes_per_modality=(
                class_conditioning_num_classes_per_modality),
            class_conditioning_embedding_dim_per_modality=(
                class_conditioning_embedding_dim_per_modality),
            class_conditioning_prepend_to_dummy_input=(
                args.class_conditioning_prepend_to_dummy_input),
            disable_start_symbol_DEBUG=args.disable_start_symbol_DEBUG,
        )

    initial_epoch = 0
    if model_checkpoint_weights is not None:
        if 'model' in model_checkpoint_weights:
            model.load_state_dict(model_checkpoint_weights['model'])
        else:
            model.load_state_dict(model_checkpoint_weights)

        if 'epoch' in model_checkpoint_weights:
            initial_weights_training_epochs = model_checkpoint_weights['epoch']
            initial_epoch = initial_weights_training_epochs + 1

    model = model.to(device)
    if args.optimizer == 'adam':
        optimizer_class = torch.optim.Adam
    elif args.optimizer == 'radam':
        optimizer_class = RAdam
    optimizer = optimizer_class(model.parameters(), lr=args.lr,
                                eps=args.optimizer_eps)

    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)

    model = model.to(device)

    MAIN_DIR = pathlib.Path(DIRPATH)
    CHECKPOINTS_DIR_PATH = (
        MAIN_DIR
        / f'checkpoints/code_prediction/vqvae-{args.vqvae_run_id}/{run_ID}/')
    if not args.disable_writes_to_disk:
        os.makedirs(CHECKPOINTS_DIR_PATH, exist_ok=True)
        with open(CHECKPOINTS_DIR_PATH / 'command_line_parameters.json', 'w') as f:
            json.dump(args.__dict__, f, indent=4)
        model.store_instantiation_parameters(
            CHECKPOINTS_DIR_PATH / 'model_instantiation_parameters.json')

    tensorboard_writer = None
    if not (args.disable_writes_to_disk or args.disable_tensorboard):
        tensorboard_dir_path = MAIN_DIR / f'runs/{run_ID}/'
        os.makedirs(tensorboard_dir_path, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_dir_path)

    scheduler = None
    if args.scheduler == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.num_epochs,
            momentum=None
        )
    elif args.scheduler == 'warmup_cosine_annealing':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.num_warmup_steps,
            num_training_steps=len(loader) * args.num_epochs
        )

    num_classes = model.n_class
    criterion = LabelSmoothingLoss(num_classes=num_classes,
                                   smoothing=args.label_smoothing,
                                   dim=1)

    checkpoint_name = f'{args.model_type}-layer_{args.hier}'
    checkpoint_path = CHECKPOINTS_DIR_PATH / f'{checkpoint_name}.pt'
    best_model_checkpoint_path = (
        CHECKPOINTS_DIR_PATH
        / f'{checkpoint_name}-best_performing.pt')

    mask_sampler = None
    if args.hier == 'top' and args.self_conditional_model:
        mask_sampler_kwargs = {
            'sequence_duration': (
                model.source_transformer_sequence_length),
            'mask_token_index': model.mask_token_index
        }
        if args.mask_sampling_strategy == 'bernoulli':
            mask_sampler = BernoulliSequenceMask(
                **mask_sampler_kwargs,
                probability=args.bernoulli_masking_probability)
        if args.mask_sampling_strategy == 'random_p_bernoulli':
            mask_sampler = UniformProbabilityBernoulliSequenceMask(
                low=args.random_p_bernoulli_p_range[0],
                high=args.random_p_bernoulli_p_range[1],
                **mask_sampler_kwargs,)
        elif args.mask_sampling_strategy == 'uniform_masked_amount':
            mask_sampler = UniformMaskedAmountSequenceMask(
                **mask_sampler_kwargs,
                min_masking_ratio=args.uniform_masked_amount_min_masking_ratio)
        elif args.mask_sampling_strategy == 'contiguous_zones':
            mask_sampler = ContiguousZonesSequenceMask(
                **mask_sampler_kwargs)

    if validation_loader is not None:
        best_validation_loss = float("inf")

    if args.evaluate_only:
        assert validation_loader is not None

        with torch.no_grad():
            total_validation_loss, total_accuracy, num_validation_samples = run_model(
                args, initial_epoch-1, validation_loader, model, optimizer,
                scheduler, device, criterion,
                tensorboard_writer=tensorboard_writer, is_training=False,
                num_codes_dictionary=model.n_class,
                mask_sampler=mask_sampler)
    else:
        for epoch in range(initial_epoch, args.num_epochs):
            run_model(args, epoch, loader, model, optimizer, scheduler, device,
                      criterion, tensorboard_writer=tensorboard_writer,
                      is_training=True,
                      mask_sampler=mask_sampler,
                      num_codes_dictionary=model.n_class,
                      clip_grad_norm=args.clip_grad_norm,
                      plot_frequency_batch=args.plot_frequency_batch,
                      drop_loss_half_DEBUG=args.drop_loss_half_DEBUG,
                      train_num_steps_sequences_DEBUG=args.train_num_steps_sequences_DEBUG)

            checkpoint_dict = {
                'command_line_arguments': args.__dict__,
                'model': model.state_dict(),
                'model_instatiation_parameters': (
                    model._instantiation_parameters),
                'epoch': epoch}
            if not args.disable_writes_to_disk:
                torch.save(checkpoint_dict, checkpoint_path)

            if validation_loader is not None:
                with torch.no_grad():
                    total_validation_loss, total_accuracy, num_validation_samples = run_model(
                        args, epoch, validation_loader, model, optimizer,
                        scheduler, device, criterion,
                        tensorboard_writer=tensorboard_writer, is_training=False,
                        num_codes_dictionary=model.n_class,
                        mask_sampler=mask_sampler,
                        train_num_steps_sequences_DEBUG=args.train_num_steps_sequences_DEBUG)
                if total_validation_loss < best_validation_loss:
                    best_validation_loss = total_validation_loss

                    validation_dict = {
                        'criterion': str(criterion),
                        'dataset': args.validation_database_path,
                        'loss': total_validation_loss
                    }
                    checkpoint_dict['validation'] = validation_dict
                    if not args.disable_writes_to_disk:
                        torch.save(checkpoint_dict, best_model_checkpoint_path)
