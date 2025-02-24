import os

# 設置目標文件夾的路徑
destination_folder = r''  # 替換為目標文件夾的路徑
mapping_file = r''

# 讀取txt對照表，儲存新舊文件名的映射關係
mapping = {}
with open(mapping_file, 'r') as f:
    for line in f:
        new_name, old_name = line.strip().split(' -> ')
        mapping[new_name] = old_name

# 遍歷目標文件夾，按照對照表重命名文件
for new_name, old_name in mapping.items():
    new_file_path = os.path.join(destination_folder, new_name)
    old_file_path = os.path.join(destination_folder, old_name)
    
    # 重命名文件
    if os.path.exists(new_file_path):
        os.rename(new_file_path, old_file_path)

print("文件已按對照表重命名。")
