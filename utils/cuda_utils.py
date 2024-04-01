from typing import List, Mapping, Tuple
import torch


def load_data_to_device(data: Mapping,
                        field_name :List[str],
                        device='cpu') -> Mapping:
    r"""
        将数据加载到对应的设
    """
    device = torch.device(device)
    if field_name is None:
        field_name = list(data.keys())
    for field in field_name:
        data[field] = data[field].to(device)
    return data[field]
