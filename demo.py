from colorized_point_cloud import *
import open3d as o3d
import matplotlib.pyplot as plt
import seaborn as sns


def show_colorized_2d_point_cloud(dataset, index, cam_number):
    o3d.visualization.draw_geometries([get_colorized_2d_point_cloud(dataset, index, cam_number)])


def show_colorized_3d_point_cloud(dataset, index, cam_number):
    o3d.visualization.draw_geometries([get_colorized_3d_point_cloud(dataset, index, cam_number)])


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


def show_colorized_point_cloud_from_two_cam(dataset, index):
    o3d.visualization.draw_geometries(get_cloud_sum_in_world_coords(dataset,
                                                                    get_colorized_3d_point_cloud(dataset, index, 2)),
                                      get_colorized_2d_point_cloud(dataset, index, 3))
