import blenderproc as bproc
import bpy
import os
import math
import h5py
import numpy as np
import shutil
from PIL import Image
from scipy.spatial.transform import Rotation
import sys
sys.append("/data/code/BlenderCollector")
try:
    from visHdf5Files import vis_data
except ModuleNotFoundError:
    from blenderproc.scripts.visHdf5Files import vis_data

data_root = "/data/code/BlenderCollector"
RESULT_SAVE_PATH = os.path.join(data_root, "output")
SCENE_PATH = os.path.join(data_root, "env_model/1/1.fbx")
OBJ_HUMAN_PATH = os.path.join(data_root, "human_model/sample00_rep00_obj")
CAMERA_POSITION_PATH = os.path.join(data_root, "camera_position/camera_positions_1")

intrinsics_path = os.path.join(RESULT_SAVE_PATH, "sparse/0/cameras.txt")
extrinsics_path = os.path.join(RESULT_SAVE_PATH, "sparse/0/images.txt")
    
def data_prepare():
    if not os.path.exists(RESULT_SAVE_PATH):
        os.makedirs(RESULT_SAVE_PATH)
    else:
        shutil.rmtree(RESULT_SAVE_PATH)
        os.makedirs(RESULT_SAVE_PATH)
    # 确保文件所在的目录存在
    os.makedirs(os.path.dirname(intrinsics_path), exist_ok=True)
    os.makedirs(os.path.dirname(extrinsics_path), exist_ok=True)
    open(intrinsics_path, 'w').close()
    open(extrinsics_path, 'w').close()
    
def hide_object(obj_pre_human):
    for obj in obj_pre_human:
        obj.blender_obj.hide_viewport = True
        obj.blender_obj.hide_render = True

def scene_init():
    bproc.init()
    obj = bproc.loader.load_obj(SCENE_PATH)
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([0, 1, 2])  # location根据自己的场景调整
    light.set_energy(1000)
    for obj in bpy.context.scene.objects:
        if obj.type == 'CAMERA':
            camera = obj
            break
    bproc.camera.set_resolution(640, 480)
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(map_by=['category_id', "instance", "name"], default_values={'category_id': 0})
    bpy.context.scene.cycles.samples = 1024
    is_look_around = True
    return camera

def save_image(data, filename):
    """将数据保存为图片"""
    if data.ndim == 2 or (data.ndim == 3 and data.shape[2] == 1):
        mode = 'L'  # 灰度图
        print("gls data:", data)
    elif data.shape[2] == 3:
        mode = 'RGB'  # RGB 图
    else:
        raise ValueError("Unsupported image format")

    # img = Image.fromarray(data.astype('uint8'), mode)
    img = Image.fromarray(data, mode)
    
    img.save(filename)
       
def blender_vis(directory, keys, line_num):
    ### 根据keys创建文件夹
    for key in keys:
        if not os.path.exists(os.path.join(directory, key)):
            os.makedirs(os.path.join(directory, key))
    """处理指定目录下的所有HDF5文件，并保存指定键值的图片"""
    for filename in os.listdir(directory):
        if filename.endswith(".hdf5"):
            file_path = os.path.join(directory, filename)
            with h5py.File(file_path, 'r') as f:
                def print_keys(name, obj):
                    print(name)  # 输出键值

                f.visititems(print_keys)  # 使用visititems遍历所有项并调用print_keys函数
                # print("f is:", f)
                for key in keys:
                    if key in f:
                        data = np.array(f[key])
                        print("data shape:", data.shape)
                        # 构建图片文件名
                        if key == "colors":
                            image_filename = f"observation_rgb_{line_num}.png"
                        elif key == "depth":
                            print("depth data:", data)
                            image_filename = f"observation_depth_{line_num}.png"
                        elif key == "category_id_segmaps" or key == "normals":
                            data = data.astype(np.float32)
                            data = np.clip(data, 0, 1)
                            image_filename = f"observation_{key}_{line_num}.png"
                        image_path = os.path.join(directory, key, image_filename)
                        vis_data(key, data, None, "", save_to_file=image_path)
                        # save_image(data, image_path)
                        print(f"Saved {image_path}")
                        import imageio
                        img = imageio.imread(image_path)
                        print(img.shape)  # 应输出 (480, 640)
                    else:
                        print(f"Key '{key}' not found in {filename}")
                line_num += 1

def matrix_world_to_colmap(matrix_world):
    # R_blender_to_colmap = np.array([
    #     [1, 0, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, -1, 0, 0],
    #     [0, 0, 0, 1]
    # ])
    # # matrix_world_blender = R_blender_to_colmap @ matrix_world @ R_blender_to_colmap
    # matrix_world_blender = R_blender_to_colmap @ matrix_world
    # 提取旋转矩阵（3x3）和平移向量（3x1）
    rotation_matrix = matrix_world[:3, :3]
    translation = matrix_world[:3, 3]
    # 将旋转矩阵转换为四元数 [w, x, y, z]
    r = Rotation.from_matrix(rotation_matrix)
    
    quaternion = r.as_quat()  # 默认顺序是 [x, y, z, w]，需调整

    # 调整四元数顺序为 [w, x, y, z]
    # quaternion = np.array([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])

    # COLMAP 外参格式：[w, x, y, z, Tx, Ty, Tz]
    colmap_extrinsics = np.concatenate([quaternion, translation])

    return colmap_extrinsics

def euler_to_rotation_matrix(angles, order='xyz'):
    """
    将欧拉角转换为旋转矩阵
    :param angles: 欧拉角，格式为 [a, b, c]，单位为弧度
    :param order: 旋转顺序，默认为 'xyz'
    :return: 旋转矩阵
    """
    a, b, c = angles
    if order == 'xyz':
        # 绕X轴旋转a
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(a), -math.sin(a)],
            [0, math.sin(a), math.cos(a)]
        ])
        # 绕Y轴旋转b
        Ry = np.array([
            [math.cos(b), 0, math.sin(b)],
            [0, 1, 0],
            [-math.sin(b), 0, math.cos(b)]
        ])
        # 绕Z轴旋转c
        Rz = np.array([
            [math.cos(c), -math.sin(c), 0],
            [math.sin(c), math.cos(c), 0],
            [0, 0, 1]
        ])
        # 组合旋转矩阵 R = Rx * Ry * Rz
        R = np.dot(Rx, np.dot(Ry, Rz))
    else:
        raise ValueError("Unsupported rotation order")
    return R

def camera_to_world_matrix(position, euler_angles, order='xyz'):
    """
    计算相机坐标系到世界坐标系的转换矩阵
    :param position: 相机在世界坐标系中的位置 [x, y, z]
    :param euler_angles: 相机的欧拉角 [a, b, c]，单位为弧度
    :param order: 欧拉角的旋转顺序，默认为 'zyx'
    :return: 4x4的转换矩阵
    """
    x, y, z = position
    R = euler_to_rotation_matrix(euler_angles, order)
    
    # 构造转换矩阵
    T = np.array([
        [R[0, 0], R[0, 1], R[0, 2], x],
        [R[1, 0], R[1, 1], R[1, 2], y],
        [R[2, 0], R[2, 1], R[2, 2], z],
        [0, 0, 0, 1]
    ])
    return T

def collect_data(camera, is_look_around, is_import_human):
    focal_length = camera.data.lens
    sensor_width = camera.data.sensor_width
    sensor_height = camera.data.sensor_height
    image_width = 640
    image_height = 480
    camera_type = "PINHOLE"
    pre_human = None
    with open(CAMERA_POSITION_PATH, "r") as f:
        if is_look_around:  ### look around代表以视角为中心，旋转360度
            line = f.readline()
            currentframe = bpy.context.scene.frame_current
            line = [float(x) for x in line.split()]
            position, euler_rotation = line[:3], line[3:6]
            print("position:", position)
            print("euler_rotation:", euler_rotation)
            line_num = 1
            for i in range(0, 361, 10):    
                euler_rotation[2] = i
                euler_rotation_rad = [math.radians(angle) for angle in euler_rotation]
                # camera2world = camera_to_world_matrix(position, euler_rotation)
                bpy.context.scene.frame_start = currentframe
                bpy.context.scene.frame_end = currentframe
                bpy.context.scene.animation_data_clear()
                # 设置帧范围
                matrix_world = bproc.math.build_transformation_mat(position, euler_rotation_rad)
                bproc.camera.add_camera_pose(matrix_world)
                cam2world = np.array(camera.matrix_world.copy())
                ### 写入内外参矩阵：
                fx = focal_length * image_width / sensor_width
                fy = focal_length * image_height / sensor_height
                cx = image_width / 2
                cy = image_height / 2
                colmap_extrinsics = matrix_world_to_colmap(cam2world)
                # 相机内参文件
                with open(intrinsics_path, "a") as camera_intrinsics_file:
                    camera_intrinsics_file.write(f"{line_num} {camera_type} {image_width} {image_height} {fx} {fy} {cx} {cy}\n")
                    camera_intrinsics_file.close()    
                # 相机外参文件
                with open(extrinsics_path, "a") as camera_extrinsics_file:
                    camera_extrinsics_file.write(f"{line_num} {colmap_extrinsics[0]} {colmap_extrinsics[1]} {colmap_extrinsics[2]} {colmap_extrinsics[3]} {colmap_extrinsics[4]} {colmap_extrinsics[5]} {colmap_extrinsics[6]} {line_num} observation_rgb_{line_num}.png\n")
                    camera_intrinsics_file.close()
                if is_import_human:
                    obj_template = os.path.join(OBJ_HUMAN_PATH, f"000{{:03d}}.obj")  # 动态生成文件名
                    if line_num > 1:
                        hide_object(pre_human)
                    current_obj_path = obj_template.format((line_num-1)*10) ### 模型编号从0开始 
                    obj_human = bproc.loader.load_obj(current_obj_path)
                    for obj in obj_human:
                        # 可以对每个物体进行操作，例如设置位置    
                        obj.set_location((-1, -2, 1))
                        # 设置欧拉旋转
                        obj.set_rotation_euler((0, 0, 0))
                    pre_human = obj_human

                data = bproc.renderer.render()
                
                # write the data to a .hdf5 container
                bproc.writer.write_hdf5(RESULT_SAVE_PATH, data)
                # 使用示例
                hdf5_directory = RESULT_SAVE_PATH  # 替换为您的HDF5文件目录
                keys = ["colors", "depth", "category_id_segmaps"]  # 要保存的键值列表 ["colors", "depth", "normals", "category_id_segmaps"]
                blender_vis(directory=hdf5_directory, keys=keys, line_num=line_num)
                line_num += 1
        else:   # 非look_around要调整camera_position，提供序列数据
            for line_num, line in enumerate(f.readlines(), start=1):
                # 清除之前的动画数据
                currentframe = bpy.context.scene.frame_current
                bpy.context.scene.frame_start = currentframe
                bpy.context.scene.frame_end = currentframe
                bpy.context.scene.animation_data_clear()
                # 设置帧范围
                line = [float(x) for x in line.split()]
                position, euler_rotation = line[:3], line[3:6]
                matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
                bproc.camera.add_camera_pose(matrix_world)
                ### 写入内外参矩阵：
                fx = focal_length * image_width / sensor_width
                fy = focal_length * image_height / sensor_height
                cx = image_width / 2
                cy = image_height / 2
                colmap_extrinsics = matrix_world_to_colmap(matrix_world)
                # 相机内参文件
                with open(intrinsics_path, "a") as camera_intrinsics_file:
                    camera_intrinsics_file.write(f"{line_num} {camera_type} {image_width} {image_height} {fx} {fy} {cx} {cy}\n")
                    camera_intrinsics_file.close()    
                # 相机外参文件
                with open(extrinsics_path, "a") as camera_extrinsics_file:
                    camera_extrinsics_file.write(f"{line_num} {colmap_extrinsics[0]} {colmap_extrinsics[1]} {colmap_extrinsics[2]} {colmap_extrinsics[3]} {colmap_extrinsics[4]} {colmap_extrinsics[5]} {line_num} observation_rgb_{line_num}.png\n")
                    camera_intrinsics_file.close()
                if is_import_human:
                    obj_template = os.path.join(OBJ_HUMAN_PATH, f"000{{:03d}}.obj")  # 动态生成文件名
                    if line_num > 1:
                        hide_object(pre_human)
                    current_obj_path = obj_template.format((line_num-1)*10) ### 模型编号从0开始 
                    obj_human = bproc.loader.load_obj(current_obj_path)
                    for obj in obj_human:
                        # 可以对每个物体进行操作，例如设置位置    
                        obj.set_location((-1, -2, 1))
                        # 设置欧拉旋转
                        obj.set_rotation_euler((0, 0, 0))
                    pre_human = obj_human

                # render the whole pipeline
                data = bproc.renderer.render()
                # write the data to a .hdf5 container
                bproc.writer.write_hdf5(RESULT_SAVE_PATH, data)
                
                # 使用示例
                hdf5_directory = RESULT_SAVE_PATH  # 替换为您的HDF5文件目录
                keys = ["colors", "depth", "category_id_segmaps"]  # 要保存的键值列表 ["colors", "depth", "normals", "category_id_segmaps"]
                blender_vis(directory=hdf5_directory, keys=keys, line_num=line_num)

def main():
    data_prepare()
    camera = scene_init()
    collect_data(camera, is_look_around=True, is_import_human=False)
    
if __name__ == '__main__':
    main()