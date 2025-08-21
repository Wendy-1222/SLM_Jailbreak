import os
import argparse
def revert_wrong_renames_in_dirs(base_path, error_suffix="_error_system_prompt_error_system_prompt", correct_suffix="_error_system_prompt"):
    """递归恢复被错误重命名的目录"""
    for dirpath, dirnames, _ in os.walk(base_path, topdown=False):
        for dirname in dirnames:
            if dirname.endswith(error_suffix):
                old_path = os.path.join(dirpath, dirname)
                new_path = os.path.join(dirpath, dirname.replace(error_suffix, correct_suffix))

                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"Restored: {old_path} -> {new_path}")
                else:
                    print(f"Skipping restore: {new_path} already exists!")

def revert_wrong_renames_in_files(base_path, error_suffix="_error_system_prompt_error_system_prompt", correct_suffix="_error_system_prompt"):
    """递归恢复被错误重命名的文件"""
    for dirpath, _, filenames in os.walk(base_path, topdown=False):
        for filename in filenames:
            base_name, ext = os.path.splitext(filename)  # 分离文件名和扩展名
            if base_name.endswith(error_suffix):
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, filename.replace(error_suffix, correct_suffix))

                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"Restored File: {old_path} -> {new_path}")
                else:
                    print(f"Skipping restore: {new_path} already exists!")


def add_suffix_dirs(base_path, suffix="_error_system_prompt"):
    """递归查找并重命名所有错误模型所在的目录"""
    for dirpath, dirnames, _ in os.walk(base_path, topdown=False):
        for dirname in dirnames:
            if dirname.startswith(args.error_model_name_prefix) and not dirname.endswith(suffix):
                old_path = os.path.join(dirpath, dirname)
                new_path = os.path.join(dirpath, dirname + suffix)

                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                else:
                    print(f"Skipping rename: {new_path} already exists!")

def add_suffix_files(base_path, suffix="_error_system_prompt"):
    """递归查找并重命名所有文件"""
    for dirpath, _, filenames in os.walk(base_path, topdown=False):  # 遍历所有文件
        for filename in filenames:
            if filename.startswith(args.error_model_name_prefix) and not filename.endswith(suffix):
                old_path = os.path.join(dirpath, filename)
                base_name, ext = os.path.splitext(old_path)  # 分离文件名和扩展名
                new_name = base_name + suffix + ext  # 重新组合
                new_path = os.path.join(dirpath, new_name)
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"Renamed File: {old_path} -> {new_path}")
                else:
                    print(f"Skipping rename: {new_path} already exists!")


def rename_files(base_path, old_filename="tinyllama-1.1B-chat-v0.6.json", new_filename="tinyllama-1.1B-chat-v0.6_error_system_prompt.json"):
    """递归重命名目录下所有指定名称的文件"""
    for dirpath, _, filenames in os.walk(base_path):
        for filename in filenames:
            if filename == old_filename:
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, new_filename)

                if not os.path.exists(new_path):  # 如果目标文件不存在
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                else:
                    print(f"Skipping rename: {new_path} already exists!")


def rename_dirs(base_path, old_name="tinyllama-1.1B-chat-v0.6", new_name="tinyllama-1.1B-chat-v0.6_error_system_prompt"):
    """递归查找并重命名指定的目录"""
    for dirpath, dirnames, _ in os.walk(base_path, topdown=False):
        for dirname in dirnames:
            if dirname == old_name:  # 检查目录名是否为目标名称
                old_path = os.path.join(dirpath, dirname)
                new_path = os.path.join(dirpath, new_name)

                if not os.path.exists(new_path):  # 如果目标目录不存在
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                else:
                    print(f"Skipping rename: {new_path} already exists!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename')
    parser.add_argument('--root_dir', type=str, default="/data2/zwh/HarmBench/results_full_50", help='Root directory')
    parser.add_argument('--error_model_name_prefix', type=str, default='smollm', help='Error model name prefix')
    parser.add_argument('--error_suffix', type=str, default="_default_system_prompt_error_system_prompt", help='Error suffix')
    parser.add_argument('--correct_suffix', type=str, default="_default_system_prompt", help='Correct suffix')
    parser.add_argument('--suffix', type=str, default="_error_system_prompt", help='Suffix')

    args = parser.parse_args()

    # 目标目录：
    subdir_target_dirs = ['HumanJailbreaks', 'PAIR', 'GCG', 'AutoPrompt', 'PEZ', 'GBDA', 'UAT']  # 修改错误的模型目录
    file_target_dirs = ['DirectRequest', 'PAP', 'HumanJailbreaks']  # 递归修改所有文件

    # 处理只需要修改文件的目录（DirectRequest 和 PAP）
    for target_dir in file_target_dirs:
        target_path = os.path.join(args.root_dir, target_dir)

        if os.path.exists(target_path) and os.path.isdir(target_path):
            print(f"Processing files recursively in: {target_path}")
            # revert_wrong_renames_in_files(target_path, error_suffix=args.error_suffix, correct_suffix=args.correct_suffix)  # 恢复错误的文件
            add_suffix_files(target_path, suffix=args.suffix)  # 添加后缀
            # rename_files(target_path, old_filename="tinyllama-1.1B-chat-v0.6.json", new_filename="tinyllama-1.1B-chat-v0.6_error_system_prompt.json")
        else:
            print(f"Directory {target_path} does not exist")

    # 处理需要修改含有错误模型子目录的目录
    for target_dir in subdir_target_dirs:
        target_path = os.path.join(args.root_dir, target_dir)

        if os.path.exists(target_path) and os.path.isdir(target_path):
            print(f"Processing directories in: {target_path}")
            # revert_wrong_renames_in_dirs(target_path, error_suffix=args.error_suffix, correct_suffix=args.correct_suffix)  # 恢复错误的目录
            add_suffix_dirs(target_path, suffix=args.suffix)  # 添加后缀
            # rename_dirs(target_path, old_name="tinyllama-1.1B-chat-v0.6", new_name="tinyllama-1.1B-chat-v0.6_error_system_prompt")
        else:
            print(f"Directory {target_path} does not exist")
