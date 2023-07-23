import open3d as o3d
import pykitti
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def show_plot_from_3d_point_cloud(points):
    x = np.asarray(points)[:, 0]
    y = np.asarray(points)[:, 1]
    z = np.asarray(points)[:, 2]
    fig = plt.figure(figsize=(32, 32))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.show()


def show_plot_from_2d_point_cloud(points, image):
    x = points[0]
    y = points[1]
    depth = points[2]
    sns.scatterplot(x=x, y=y, hue=depth, alpha=0.5, s=2, legend=False)
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)
    plt.show()


def show_plot_from_2d_point_cloud_with_colors(points, pixels, image):
    colors = transform_pixels_for_point_cloud(points, pixels, image)
    x = points[0]
    y = points[1]
    fig, ax = plt.subplots()
    sns.scatterplot(x=x, y=y, color=colors, alpha=1, s=2, legend=False, ax=ax)
    ax.set_aspect('equal')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    plt.xticks([])
    plt.yticks([])
    plt.show()


def visualize_2d_point_cloud(dataset, velo_data, image, cam_number):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(velo_data[:, :3])
    if cam_number == 2:
        point_cloud.transform(dataset.calib.T_cam2_velo)
        points3d_coordinates_in_2d = dataset.calib.K_cam2 @ np.asarray(point_cloud.points).transpose()
    else:
        point_cloud.transform(dataset.calib.T_cam3_velo)
        points3d_coordinates_in_2d = dataset.calib.K_cam3 @ np.asarray(point_cloud.points).transpose()
    points3d_coordinates_in_2d = points3d_coordinates_in_2d[:, points3d_coordinates_in_2d[0] >= 0]
    points3d_coordinates_in_2d = points3d_coordinates_in_2d[:, points3d_coordinates_in_2d[1] >= 0]
    points3d_coordinates_in_2d = points3d_coordinates_in_2d[:, points3d_coordinates_in_2d[2] >= 0]
    points_depth = points3d_coordinates_in_2d[2]
    points3d_coordinates_in_2d = points3d_coordinates_in_2d[:2, :] / points3d_coordinates_in_2d[2, :]
    points3d_coordinates_in_2d = np.vstack((points3d_coordinates_in_2d, points_depth))
    points3d_coordinates_in_2d = points3d_coordinates_in_2d[:, points3d_coordinates_in_2d[0] <= image.width]
    points3d_coordinates_in_2d = points3d_coordinates_in_2d[:, points3d_coordinates_in_2d[1] <= image.height]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points3d_coordinates_in_2d.transpose())
    point_cloud.colors = o3d.utility.Vector3dVector(transform_pixels_for_point_cloud(points3d_coordinates_in_2d,
                                                                                     list(image.getdata()), image))
    o3d.visualization.draw_geometries([point_cloud])


def transform_pixels_for_point_cloud(points, pixels, image):
    colors = []
    for i in range(points.shape[1]):
        colors.append(pixels[(int(points[1, i]) * image.width + int(points[0, i]))])
    for i in range(len(colors)):
        colors[i] = (colors[i][0]/255, colors[i][1]/255, colors[i][2]/255)
    return colors


# основной код, который красит точки в цвета пикселей
def get_colored_point_cloud(dataset, index, cam_number):
    velo_data = dataset.get_velo(index)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(velo_data[:, :3])
    if cam_number == 2:
        image = dataset.get_cam2(index)
        point_cloud.transform(dataset.calib.T_cam2_velo)
    else:
        image = dataset.get_cam3(index)
        point_cloud.transform(dataset.calib.T_cam3_velo)
    points3d_coordinates_in_2d = dataset.calib.K_cam3 @ np.asarray(point_cloud.points).transpose()
    points3d_coordinates_in_2d = points3d_coordinates_in_2d[:, points3d_coordinates_in_2d[0] >= 0]
    points3d_coordinates_in_2d = points3d_coordinates_in_2d[:, points3d_coordinates_in_2d[1] >= 0]
    points3d_coordinates_in_2d = points3d_coordinates_in_2d[:, points3d_coordinates_in_2d[2] >= 0]
    points_depth = points3d_coordinates_in_2d[2]
    points3d_coordinates_in_2d = points3d_coordinates_in_2d[:2, :] / points3d_coordinates_in_2d[2, :]
    points3d_coordinates_in_2d = np.vstack((points3d_coordinates_in_2d, points_depth))
    points3d_coordinates_in_2d = points3d_coordinates_in_2d[:, points3d_coordinates_in_2d[0] <= image.width]
    points3d_coordinates_in_2d = points3d_coordinates_in_2d[:, points3d_coordinates_in_2d[1] <= image.height]
    points_depth = points3d_coordinates_in_2d[2]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.colors = o3d.utility.Vector3dVector(
        transform_pixels_for_point_cloud(points3d_coordinates_in_2d, list(image.getdata()), image))
    points3d_coordinates_in_2d = points3d_coordinates_in_2d[:2, :] * points3d_coordinates_in_2d[2, :]
    points3d_coordinates_in_2d = np.vstack((points3d_coordinates_in_2d, points_depth))
    if cam_number == 2:
        new_points = np.linalg.inv(dataset.calib.K_cam2) @ points3d_coordinates_in_2d
    else:
        new_points = np.linalg.inv(dataset.calib.K_cam3) @ points3d_coordinates_in_2d
    point_cloud.points = o3d.utility.Vector3dVector(new_points.transpose())
    return point_cloud


def get_cloud_sum_in_world_coords(cloud_2, cloud_3):
    cloud_2.transform(np.linalg.inv(data.calib.T_cam2_velo)).transform(data.calib.T_cam0_velo).transform(data.poses[0])
    cloud_3.transform(np.linalg.inv(data.calib.T_cam3_velo)).transform(data.calib.T_cam0_velo).transform(data.poses[0])
    cloud_3 += cloud_2
    return cloud_3


path_to_dataset = r''
sequence = '00'
data = pykitti.odometry(path_to_dataset, sequence, frames=range(0, 10, 2))

cloud = get_cloud_sum_in_world_coords(get_colored_point_cloud(data, 2, 2), get_colored_point_cloud(data, 2, 3))
o3d.visualization.draw_geometries([cloud])
