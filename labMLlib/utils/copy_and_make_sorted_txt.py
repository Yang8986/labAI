import os
import shutil

# 設置源文件夾和目標文件夾的路径
source_folder = r''  # 替换為源文件夾的路徑
destination_folder = r''  # 替換為目標文件夾的路径
mapping_file = r''

# 創建目標文件夾（如果不存在）
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 獲取源文件夹中的所有文件并排序
files = sorted(os.listdir(source_folder))

# 創建txt對照表並重命名文件
with open(mapping_file, 'w') as f:
    for i, file_name in enumerate(files, 1):
        old_file_path = os.path.join(source_folder, file_name)
        new_file_name = f"{i}{os.path.splitext(file_name)[1]}"  # 新文件名為1, 2, 3...，並保留擴展名
        new_file_path = os.path.join(destination_folder, new_file_name)
        
        # 複製文件到目標文件並重命名
        shutil.copy2(old_file_path, new_file_path)
        
        # 寫入對照表
        f.write(f"{new_file_name} -> {file_name}\n")

print("文件已成功複製並重命名，對照表已生成。")
