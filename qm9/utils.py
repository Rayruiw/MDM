import torch


def _get_num_heavy_atoms(x):
    """
    Return heavy atom count as a float tensor of shape [B] or [1].

    Priority:
    1) use existing `num_heavy_atoms`
    2) fallback: infer from `atom_type` by counting atoms with Z != 1
    """
    if hasattr(x, 'num_heavy_atoms'):
        n_heavy = x.num_heavy_atoms
        if not torch.is_tensor(n_heavy):
            n_heavy = torch.tensor(n_heavy, dtype=torch.float32)
        return n_heavy.view(-1).float()

    if hasattr(x, 'atom_type'):
        atom_type = x.atom_type
        if not torch.is_tensor(atom_type):
            atom_type = torch.tensor(atom_type)
        return (atom_type.view(-1) != 1).sum().view(1).float()

    raise AttributeError("Object has neither 'num_heavy_atoms' nor 'atom_type'.")


def _get_property_value(data, property_key):
    """
    Read a graph-level scalar property from a single Data object.

    Returns a tensor of shape [1] or [K], but for current QM9 conditional
    generation usage we mainly expect scalar outputs.
    """
    if property_key == 'alpha_per_heavy':
        n_heavy = _get_num_heavy_atoms(data)
        return data.alpha.view(-1).float() / n_heavy

    elif property_key == 'Cv_per_heavy':
        n_heavy = _get_num_heavy_atoms(data)
        return data.Cv.view(-1).float() / n_heavy

    else:
        v = getattr(data, property_key)
        if not torch.is_tensor(v):
            v = torch.tensor(v, dtype=torch.float32)
        return v.view(-1).float()


def compute_mean_mad(dataset, properties, dataset_name):
    if dataset_name == 'qm9' or dataset_name == 'qm9_second_half':
        return compute_mean_mad_from_dataloader(dataset, properties)
    else:
        raise Exception('Wrong dataset name')


def compute_mean_mad_from_dataloader(dataset, properties):
    """
    Compute mean / MAD from the actual dataset subset, not from dataset.data.
    This is critical when using second-half splits or any sliced dataset.
    """
    property_norms = {}

    for property_key in properties:
        values = []
        for i in range(len(dataset)):
            values.append(_get_property_value(dataset[i], property_key))

        values = torch.cat(values, dim=0)
        mean = torch.mean(values)
        mad = torch.mean(torch.abs(values - mean))
        max_value = torch.max(values)
        min_value = torch.min(values)

        if mad.item() == 0:
            mad = torch.tensor(1.0, dtype=values.dtype)

        property_norms[property_key] = {
            'mean': mean,
            'mad': mad,
            'max': max_value,
            'min': min_value,
        }

    return property_norms


edges_dic = {}


def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx * n_nodes)
                        cols.append(j + batch_idx * n_nodes)
    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges


def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars


def prepare_context(conditioning, minibatch, property_norms):
    """
    Build node-level conditioning tensor of shape [N, C].

    This matches the current MDM.py interface where context is concatenated
    with node/atom features, so final returned context must be node-level.

    For graph-level QM9 scalar properties:
      graph-level: [B, 1]
      node-level : [N, 1] via minibatch.batch broadcast
    """
    context_node_nf = 0
    context_list = []

    batch_index = minibatch.batch   # [N]
    num_nodes = batch_index.size(0)

    for key in conditioning:
        # ------------------------------------------------------------
        # 1) Build graph-level raw property tensor [B, 1]
        # ------------------------------------------------------------
        if key == 'alpha_per_heavy':
            n_heavy = _get_num_heavy_atoms(minibatch).to(batch_index.device).view(-1, 1)
            properties = minibatch.alpha.to(batch_index.device).float().view(-1, 1) / n_heavy

        elif key == 'Cv_per_heavy':
            n_heavy = _get_num_heavy_atoms(minibatch).to(batch_index.device).view(-1, 1)
            properties = minibatch.Cv.to(batch_index.device).float().view(-1, 1) / n_heavy

        else:
            properties = getattr(minibatch, key)

            if not torch.is_tensor(properties):
                properties = torch.tensor(
                    properties, dtype=torch.float32, device=batch_index.device
                )
            else:
                properties = properties.to(batch_index.device).float()

            # Standardize graph-level scalar shape to [B, 1]
            if properties.dim() == 0:
                properties = properties.view(1, 1)
            elif properties.dim() == 1:
                properties = properties.view(-1, 1)

        # ------------------------------------------------------------
        # 2) Normalize at graph level
        # ------------------------------------------------------------
        mean = property_norms[key]['mean']
        mad = property_norms[key]['mad']

        if torch.is_tensor(mean):
            mean = mean.to(properties.device, dtype=properties.dtype)
        else:
            mean = torch.tensor(mean, device=properties.device, dtype=properties.dtype)

        if torch.is_tensor(mad):
            mad = mad.to(properties.device, dtype=properties.dtype)
        else:
            mad = torch.tensor(mad, device=properties.device, dtype=properties.dtype)

        if mad.item() == 0:
            mad = torch.tensor(1.0, device=properties.device, dtype=properties.dtype)

        properties = (properties - mean) / mad   # [B, C]

        # ------------------------------------------------------------
        # 3) Broadcast graph-level property to node-level context
        # ------------------------------------------------------------
        if properties.dim() == 2:
            # [B, C] -> [N, C]
            properties = properties.index_select(0, batch_index)
            context_list.append(properties)
            context_node_nf += properties.size(1)

        elif properties.dim() == 3:
            # [B, A, C] -> [N, A*C]
            properties = properties.index_select(0, batch_index)
            properties = properties.view(num_nodes, -1)
            context_list.append(properties)
            context_node_nf += properties.size(1)

        else:
            raise ValueError(f'Invalid tensor size for context key {key}: {properties.size()}')

    context = torch.cat(context_list, dim=1)

    assert context.dim() == 2, f"context must be 2D, got {context.shape}"
    assert context.size(0) == num_nodes, \
        f"context first dim {context.size(0)} != num_nodes {num_nodes}"
    assert context.size(1) == context_node_nf, \
        f"context feature dim {context.size(1)} != expected {context_node_nf}"

    return context