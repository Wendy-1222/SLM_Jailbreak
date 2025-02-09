import os

# 设置根目录和需要修改的目录
root_dir = "/data2/zwh/HarmBench/results"

# 目标目录：
subdir_target_dirs = ['HumanJailbreaks', 'PAIR', 'GCG', 'AutoPrompt', 'PEZ', 'GBDA', 'UAT']  # 修改 tinyllama_ 目录
file_target_dirs = ['DirectRequest', 'PAP', 'HumanJailbreaks']  # 递归修改所有文件

def revert_wrong_renames_in_dirs(base_path):
    """递归恢复被错误重命名的目录"""
    for dirpath, dirnames, _ in os.walk(base_path, topdown=False):
        for dirname in dirnames:
            if dirname.endswith("_error_system_prompt_error_system_prompt"):
                old_path = os.path.join(dirpath, dirname)
                correct_name = dirname.replace("_error_system_prompt_error_system_prompt", "_error_system_prompt")
                new_path = os.path.join(dirpath, correct_name)

                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"Restored: {old_path} -> {new_path}")
                else:
                    print(f"Skipping restore: {new_path} already exists!")

def rename_tinyllama_dirs(base_path):
    """递归查找并重命名所有以 'tinyllama_' 开头的目录"""
    for dirpath, dirnames, _ in os.walk(base_path, topdown=False):
        for dirname in dirnames:
            if dirname.startswith("tinyllama") and not dirname.endswith("_error_system_prompt"):
                old_path = os.path.join(dirpath, dirname)
                new_path = os.path.join(dirpath, dirname + "_error_system_prompt")

                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                else:
                    print(f"Skipping rename: {new_path} already exists!")

def rename_files_recursively(base_path):
    """递归查找并重命名所有文件"""
    for dirpath, _, filenames in os.walk(base_path, topdown=False):  # 遍历所有文件
        for filename in filenames:
            # if filename.endswith("_error_system_prompt"):
            #     old_path = os.path.join(dirpath, filename)
            #     new_path = os.path.join(dirpath, filename.replace("_error_system_prompt", ""))
            #     try:
            #         os.rename(old_path, new_path)
            #         print(f"重命名: {old_path} -> {new_path}")
            #     except Exception as e:
            #         print(f"无法重命名 {old_path}: {e}")
            if filename.startswith("tinyllama") and not filename.endswith("_error_system_prompt"):
                old_path = os.path.join(dirpath, filename)
                base_name, ext = os.path.splitext(old_path)  # 分离文件名和扩展名
                new_name = f"{base_name}_error_system_prompt{ext}"  # 重新组合
                new_path = os.path.join(dirpath, new_name)
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"Renamed File: {old_path} -> {new_path}")
                else:
                    print(f"Skipping rename: {new_path} already exists!")

# 处理只需要修改文件的目录（DirectRequest 和 PAP）
for target_dir in file_target_dirs:
    target_path = os.path.join(root_dir, target_dir)

    if os.path.exists(target_path) and os.path.isdir(target_path):
        print(f"Processing files recursively in: {target_path}")
        rename_files_recursively(target_path)
    else:
        print(f"Directory {target_path} does not exist")

# 处理需要修改 `tinyllama_` 目录的其他目标目录
for target_dir in subdir_target_dirs:
    target_path = os.path.join(root_dir, target_dir)

    if os.path.exists(target_path) and os.path.isdir(target_path):
        print(f"Processing directories in: {target_path}")
        revert_wrong_renames_in_dirs(target_path)  # 先恢复错误的目录
        rename_tinyllama_dirs(target_path)  # 再进行正确的重命名
    else:
        print(f"Directory {target_path} does not exist")
