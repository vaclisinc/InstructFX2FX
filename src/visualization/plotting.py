import numpy as np
import torch
import inspect
import plotly.graph_objects as go
from sklearn.manifold import TSNE


def plot_embedding_space_tsne(audio_embeddings, text_embeddings={},
                               perplexity=30, random_state=42,
                               title="Embedding Space (t-SNE)",
                               n_components=2):
    """Visualize audio and text embeddings using t-SNE with Plotly.

    Args:
        audio_embeddings: List of audio embedding tensors (e.g., [emb1, emb2, emb3])
                          Will use variable names from caller's scope
        text_embeddings: Dict with text labels as keys and embedding tensors as values
                         (e.g., {"bright": emb_bright, "dark": emb_dark})
        perplexity: t-SNE perplexity parameter (default 30)
        random_state: Random seed for reproducibility
        title: Plot title
        n_components: 2 for 2D plot, 3 for 3D plot (default 2)

    Returns:
        fig: Plotly figure object (interactive, call fig.show() to display)
    """
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3")

    perplexity = min(perplexity, len(audio_embeddings) + len(text_embeddings) - 1)

    # Get variable names for audio embeddings from caller's scope
    frame = inspect.currentframe().f_back
    caller_locals = frame.f_locals
    id_to_name = {id(v): k for k, v in caller_locals.items()}

    audio_names = [id_to_name.get(id(emb), f"audio_{i}") for i, emb in enumerate(audio_embeddings)]

    # Convert embeddings to numpy
    def to_2d_np(emb):
        arr = emb.detach().cpu().numpy() if isinstance(emb, torch.Tensor) else np.asarray(emb)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    audio_emb_np = [to_2d_np(e) for e in audio_embeddings]
    text_emb_np  = {label: to_2d_np(e) for label, e in text_embeddings.items()}

    # Combine all embeddings for t-SNE
    all_embeddings, all_labels, embedding_types = [], [], []

    for name, emb in zip(audio_names, audio_emb_np):
        all_embeddings.append(emb[0])
        all_labels.append(name)
        embedding_types.append('Audio')

    for label, emb in text_emb_np.items():
        all_embeddings.append(emb[0])
        all_labels.append(label)
        embedding_types.append('Text')

    all_embeddings = np.array(all_embeddings)
    embedding_types = np.array(embedding_types)

    # Apply t-SNE
    print(f"🔄 Applying {n_components}D t-SNE to {len(all_embeddings)} embeddings...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=random_state, verbose=1)
    r = tsne.fit_transform(all_embeddings)
    print("✓ t-SNE completed")

    colors = {'Audio': '#4A90D9', 'Text': '#E8794A'}
    symbols = {'Audio': 'circle', 'Text': 'diamond'}

    fig = go.Figure()

    for group in ('Audio', 'Text'):
        mask = embedding_types == group
        if not np.any(mask):
            continue

        labels = [all_labels[i] for i in np.where(mask)[0]]
        kwargs = dict(
            name=group,
            mode='markers+text',
            text=labels,
            textposition='top center',
            marker=dict(
                size=12,
                color=colors[group],
                symbol=symbols[group],
                line=dict(width=1.5, color='black'),
                opacity=0.85,
            ),
            hovertemplate='<b>%{text}</b><extra></extra>',
        )

        if n_components == 2:
            fig.add_trace(go.Scatter(
                x=r[mask, 0], y=r[mask, 1],
                **kwargs
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=r[mask, 0], y=r[mask, 1], z=r[mask, 2],
                **kwargs
            ))

    layout_common = dict(
        title=dict(text=title, font=dict(size=16)),
        legend=dict(font=dict(size=12)),
        template='plotly_white',
    )

    if n_components == 2:
        fig.update_layout(
            **layout_common,
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
        )
    else:
        fig.update_layout(
            **layout_common,
            scene=dict(
                xaxis_title='t-SNE 1',
                yaxis_title='t-SNE 2',
                zaxis_title='t-SNE 3',
            ),
        )

    return fig
