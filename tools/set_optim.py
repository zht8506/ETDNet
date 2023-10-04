from torch import optim as optim
import datetime

def build_optimizer_swin(model,lr=1e-4,weight_decay=0.05,dir_path=None):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip_keywords = {}
    if dir_path is not None:
        with open(dir_path,"w") as f:
            f.write("时间" + " " + datetime.datetime.now().strftime('%m-%d %H:%M') + "\n")
    if hasattr(model.backbone.backbone, 'no_weight_decay_keywords'):
        skip_keywords = model.backbone.backbone.no_weight_decay_keywords()
    else:
        skip_keywords = {'relative_position_bias_table'}
    parameters = set_weight_decay(model, skip_keywords,dir_path)

    optimizer = optim.AdamW(parameters,
                            betas=(0.9, 0.999),
                            lr=lr,
                            weight_decay=weight_decay)

    return optimizer

def build_optimizer_lightvit(model,lr=1e-4,weight_decay=0.05,dir_path=None):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    if dir_path is not None:
        with open(dir_path,"w") as f:
            f.write("时间" + " " + datetime.datetime.now().strftime('%m-%d %H:%M') + "\n")
    skip_keywords = {}
    parameters = set_weight_decay(model, skip_keywords,dir_path)

    optimizer = optim.AdamW(parameters,
                            betas=(0.9, 0.999),
                            lr=lr,
                            weight_decay=weight_decay)

    return optimizer

def set_weight_decay(model, skip_keywords=(),dir_path=None):
    has_decay = []
    no_decay = []
    if dir_path is not None:
        with open(dir_path, "a") as f:
            f.write(str(skip_keywords))
            f.write("\n")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # logger.info(f"{name} has no weight decay")
            if dir_path is not None:
                with open(dir_path,"a") as f:
                    f.write(f"{name} has no weight decay"+"\n")
            print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin