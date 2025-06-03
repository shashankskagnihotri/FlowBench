import itertools
from typing import Dict

fgsm_arguments = [
    "attack",
    "attack_norm",
    "attack_epsilon",
    "attack_targeted",
    "attack_target",
    "attack_loss",
    "attack_optim_target",
]
bim_pgd_cospgd_arguments = [
    "attack",
    "attack_norm",
    "attack_epsilon",
    "attack_alpha",
    "attack_targeted",
    "attack_target",
    "attack_loss",
    "attack_iterations",
    "attack_optim_target",
]
apgd_arguments = [
    "attack",
    "attack_norm",
    "attack_epsilon",
    "attack_targeted",
    "attack_target",
    "attack_loss",
    "apgd_rho",
    "apgd_n_restarts",
    "apgd_eot_iter",
    "apgd_seed",
    "apgd_steps",
]
weather_arguments = [
    "attack",
    "attack_targeted",
    "attack_target",
    "attack_loss",
    "weather_optimizer",
    "weather_steps",
    "weather_learn_offset",
    "weather_learn_motionoffset",
    "weather_learn_color",
    "weather_learn_transparency",
    "weather_alph_motion",
    "weather_alph_motionoffset",
    "weather_data",
    "weather_dataset",
    "weather_dataset_stage",
    "weather_rendering_method",
    "weather_transparency_scale",
    "weather_depth_check",
    "weather_depth_check_differentiable",
    "weather_scene_scale",
    "weather_recolor",
    "weather_do_motionblur",
    "weather_motionblur_scale",
    "weather_motionblur_samples",
    "weather_model_iters",
    "weather_flakesize_max",
    "weather_depth_decay",
    "weather_constant_transparency",
    "weather_motion_y",
    "weather_do_motionblur",
    "weather_flake_r",
    "weather_flake_g",
    "weather_flake_b",
    "weather_motion_random_scale",
    "weather_motion_random_angle",
    "weather_flake_random_h",
    "weather_flake_random_l",
    "weather_frame_per_scene",
    "weather_no_flake_dat",
    "weather_lr",
    "weather_unregistered_artifacts",
    "weather_num_flakes",
]
pcfa_arguments = [
    "attack",
    "attack_targeted",
    "attack_target",
    "attack_loss",
    "attack_epsilon",
    "attack_alpha",
    "attack_iterations",
    "pcfa_boxconstraint",
]
tdcc_arguments = [
    "attack",
    "tdcc_corruption",
    "tdcc_intensity",
    "attack_targeted",
]
cc_arguments = [
    "attack",
    "attack_targeted",
    "cc_name",
    "cc_severity",
]
no_attack_arguments = ["attack", "attack_targeted"]
targeted_arguments = ["attack_target"]
untargeted_arguments = ["attack_optim_target"]


class AttackArgumentParser:
    def __init__(self, args) -> None:
        self.attack_args = {}
        self.argument_lists = []
        self.number_of_args = 0
        self.index = 0
        for arg in vars(args):
            if (
                arg.startswith("attack")
                or arg.startswith("pcfa")
                or arg.startswith("apgd")
                or arg.startswith("3dcc")
                or arg.startswith("tdcc")
                or arg.startswith("cc")
                or arg.startswith("weather")
            ):
                self.attack_args[arg] = list(set(self.to_list(getattr(args, arg))))
        self.number_of_args = len(self.attack_args.keys())
        self.args_list_to_arg_sets()
        self.filter_arguments()

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.argument_lists):
            raise StopIteration
        else:
            argument_dic = self.argument_lists[self.index]
            self.index = self.index + 1
            return argument_dic

    def args_list_to_arg_sets(self):
        args_list = []
        for arg_list in self.attack_args.values():
            args_list.append(arg_list)
        args_list_flat = self.flatten_and_combine(args_list)

        args_list = [
            args_list_flat[i : i + self.number_of_args]
            for i in range(0, len(args_list_flat), self.number_of_args)
        ]
        for entry in args_list:
            arguments = {}
            counter = 0
            for arg_name in self.attack_args.keys():
                arguments[arg_name] = entry[counter]
                counter = counter + 1
            self.argument_lists.append(arguments)

    def flatten_and_combine(self, lists):
        if len(lists) == 1:
            return lists[0]
        else:
            # Generate all combinations from sublists first
            rest_combined = list(itertools.product(*lists[1:]))
            result = []
            for item in lists[0]:
                for combo in rest_combined:
                    result.extend([item] + list(combo))
            return result

    def to_list(self, var):
        if not isinstance(var, list):
            return [var]
        return var

    def filter_arguments(self):
        for i in range(0, len(self.argument_lists)):
            entry = self.argument_lists[i]
            attack = entry["attack"]
            targeted = entry["attack_targeted"]
            to_remove = dict(entry)

            match attack:  # Commit adversarial attack
                case "fgsm":
                    for arg_name in entry.keys():
                        if arg_name not in fgsm_arguments:
                            del to_remove[arg_name]
                    if targeted == False:
                        for targeted_args in targeted_arguments:
                            del to_remove[targeted_args]
                    else:
                        for untargeted_args in untargeted_arguments:
                            del to_remove[untargeted_args]
                case "pgd" | "cospgd" | "bim":
                    for arg_name in entry.keys():
                        if arg_name not in bim_pgd_cospgd_arguments:
                            del to_remove[arg_name]
                    if targeted == False:
                        for targeted_args in targeted_arguments:
                            del to_remove[targeted_args]
                    else:
                        for untargeted_args in untargeted_arguments:
                            del to_remove[untargeted_args]
                case "apgd":
                    for arg_name in entry.keys():
                        if arg_name not in apgd_arguments:
                            del to_remove[arg_name]
                    if targeted == False:
                        for targeted_args in targeted_arguments:
                            del to_remove[targeted_args]
                case "pcfa":
                    for arg_name in entry.keys():
                        if arg_name not in pcfa_arguments:
                            del to_remove[arg_name]
                case "3dcc":
                    for arg_name in entry.keys():
                        if arg_name not in tdcc_arguments:
                            del to_remove[arg_name]
                        elif arg_name == "attack_targeted":
                            to_remove["attack_targeted"] = False
                case "common_corruptions":
                    for arg_name in entry.keys():
                        if arg_name not in cc_arguments:
                            del to_remove[arg_name]
                        elif arg_name == "attack_targeted":
                            to_remove["attack_targeted"] = False
                case "none":
                    for arg_name in entry.keys():
                        if arg_name not in no_attack_arguments:
                            del to_remove[arg_name]
                        elif arg_name == "attack_targeted":
                            to_remove["attack_targeted"] = False
                case "weather":
                    for arg_name in entry.keys():
                        if arg_name not in weather_arguments:
                            del to_remove[arg_name]
            self.argument_lists[i] = to_remove

        indexes_to_remove = set()
        for i in range(0, len(self.argument_lists)):
            if i in indexes_to_remove:
                continue
            for j in range(i + 1, len(self.argument_lists)):
                if self.argument_lists[i] == self.argument_lists[j]:
                    indexes_to_remove.add(j)

        new_argument_list = []
        for i in range(0, len(self.argument_lists)):
            if i not in indexes_to_remove:
                new_argument_list.append(self.argument_lists[i])
        self.argument_lists = new_argument_list
        self.filter_pcfa_untargeted()
        self.tdcc_to_the_end()

    def filter_pcfa_untargeted(self):
        pcfa_targeted_flag = False
        pcfa_untargeted_indices = []
        for i in range(0, len(self.argument_lists)):
            if not self.argument_lists[i]["attack"] == "pcfa":
                continue
            else:
                if not self.argument_lists[i]["attack_targeted"]:
                    pcfa_untargeted_indices.append(i)
                else:
                    pcfa_targeted_flag = (
                        pcfa_targeted_flag or self.argument_lists[i]["attack_targeted"]
                    )
        if pcfa_targeted_flag and len(pcfa_untargeted_indices) > 0:
            new_argument_list = []
            for i in range(0, len(self.argument_lists)):
                if i not in pcfa_untargeted_indices:
                    new_argument_list.append(self.argument_lists[i])
            self.argument_lists = new_argument_list

    def tdcc_to_the_end(self):
        tdcc_indeces = []
        tdcc_entries = []
        new_argument_list = []
        for i in range(0, len(self.argument_lists)):
            if self.argument_lists[i]["attack"] == "3dcc":
                tdcc_indeces.append(i)
                tdcc_entries.append(self.argument_lists[i])
        if len(tdcc_indeces) > 0:
            for i in range(0, len(self.argument_lists)):
                if i not in tdcc_indeces:
                    new_argument_list.append(self.argument_lists[i])
            new_argument_list.extend(tdcc_entries)
            self.argument_lists = new_argument_list


def attack_targeted_string(s):
    if s.upper() in {"TRUE", "1"}:
        return True
    elif s.upper() in {"FALSE", "0"}:
        return False
    else:
        raise ValueError("Not a valid boolean string")


def attack_arg_string(attack_args: Dict[str, object]):
    string = ""
    if attack_args["attack"] == "none":
        string = string + "no attack"
    else:
        string = "|"
        for key, value in attack_args.items():
            if isinstance(value, float):
                value = round(value, 2)
            string = string + key + ":" + str(value) + "|"
    return string.strip()
