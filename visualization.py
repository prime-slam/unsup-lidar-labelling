import colorized_point_cloud
import open3d as o3d
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def show_colorized_2d_point_cloud(dataset, index_of_image, cam_number):
    o3d.visualization.draw_geometries([colorized_point_cloud.
                                      get_colorized_2d_point_cloud(dataset, index_of_image, cam_number)])


def show_colorized_3d_point_cloud(dataset, index_of_image, cam_number):
    o3d.visualization.draw_geometries([colorized_point_cloud.
                                      get_colorized_3d_point_cloud(dataset, index_of_image, cam_number)])


def visualize_3d_point_cloud_on_plot(points):
    x = np.asarray(points)[:, 0]
    y = np.asarray(points)[:, 1]
    z = np.asarray(points)[:, 2]
    fig = plt.figure(figsize=(32, 32))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.show()


def visualize_2d_point_cloud_on_plot(points, image):
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


def visualize_2d_colorized_point_cloud_on_plot(points, image):
    colors = colorized_point_cloud.get_list_of_point_colors_from_image(points, image)
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


def show_colorized_point_cloud_from_two_cam(dataset, index):
    o3d.visualization.draw_geometries(colorized_point_cloud.get_cloud_union_in_world_coords(dataset,
                                      colorized_point_cloud.get_colorized_3d_point_cloud(dataset, index, 2),
                                      colorized_point_cloud.get_colorized_2d_point_cloud(dataset, index, 3)))