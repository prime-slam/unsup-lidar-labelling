# Copyright (c) 2023, Amarskiy Artem and Yaroslav Muravev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import seaborn as sns
import colorized_point_cloud


def show_colorized_2d_point_cloud(dataset, index_of_image, cam_number):
    """
    Show a colorized 2D point cloud based on the Velodyne data and the given image.

    Parameters:
        dataset (pykitti.odometry): The dataset containing the Velodyne and camera data.
        index_of_image (int): Index of the image in the dataset.
        cam_number (int): Camera number (2 for camera 2, 3 for camera 3).

    Returns:
        None
    """
    o3d.visualization.draw_geometries(
        [
            colorized_point_cloud.get_colorized_2d_point_cloud(
                dataset, index_of_image, cam_number
            )
        ]
    )


def show_colorized_3d_point_cloud(dataset, index_of_image, cam_number):
    """
    Show a colorized 3D point cloud based on the Velodyne data and the given image.

    Parameters:
        dataset (pykitti.odometry): The dataset containing the Velodyne and camera data.
        index_of_image (int): Index of the image in the dataset.
        cam_number (int): Camera number (2 for camera 2, 3 for camera 3).

    Returns:
        None
    """
    o3d.visualization.draw_geometries(
        [
            colorized_point_cloud.get_colorized_3d_point_cloud(
                dataset, index_of_image, cam_number
            )
        ]
    )


def visualize_3d_point_cloud_on_plot(points, subplot_position=111, figsize=(32, 32)):
    """
    Visualize a 3D point cloud on a 3D plot.

    Parameters:
        points (numpy.ndarray): Array of 3D points (shape: (N, 3)).
        subplot_position (int): Position of the subplot.
        figsize (tuple): Figure size.

    Returns:
        None
    """
    x = np.asarray(points)[:, 0]
    y = np.asarray(points)[:, 1]
    z = np.asarray(points)[:, 2]
    fig = plt.figure(figsize)
    ax = fig.add_subplot(subplot_position, projection="3d")
    ax.scatter(x, y, z)
    plt.show()


def visualize_2d_point_cloud_on_plot(points, image, point_square=2, transparency=0.5):
    """
    Visualize a 2D point cloud on a 2D plot overlaid on the given image.

    Parameters:
        points (numpy.ndarray): Array of 2D points (shape: (3, N)).
        image (PIL.Image): Image object for visualization.
        point_square (int): Size of the plotted points.
        transparency (float): Alpha value for the plotted points.

    Returns:
        None
    """
    x = points[0]
    y = points[1]
    depth = points[2]
    sns.scatterplot(
        x=x, y=y, hue=depth, alpha=transparency, s=point_square, legend=False
    )
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)
    plt.show()


def visualize_2d_colorized_point_cloud_on_plot(
    points, image, point_square=2, transparency=0.5
):
    """
    Visualize a 2D colorized point cloud on a 2D plot overlaid on the given image.

    Parameters:
        points (numpy.ndarray): Array of 2D points (shape: (3, N)).
        image (PIL.Image): Image object
        point_square (int): Size of the plotted points.
        transparency (float): Alpha value for the plotted points.

    Returns:
        None
    """
    colors = colorized_point_cloud.get_list_of_point_colors_from_image(points, image)
    x = points[0]
    y = points[1]
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=x, y=y, color=colors, alpha=transparency, s=point_square, legend=False, ax=ax
    )
    ax.set_aspect("equal")
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    plt.xticks([])
    plt.yticks([])
    plt.show()


def show_colorized_point_cloud_from_two_cam(dataset, index):
    """
    Show the union of colorized 3D point clouds from camera 2 and camera 3.

    Parameters:
        dataset (pykitti.odometry): The dataset containing the Velodyne and camera data.
        index (int): Index of the velodyne data in the dataset.

    Returns:
        None
    """
    o3d.visualization.draw_geometries(
        colorized_point_cloud.get_cloud_union_in_world_coords(
            dataset,
            colorized_point_cloud.get_colorized_3d_point_cloud(dataset, index, 2),
            colorized_point_cloud.get_colorized_2d_point_cloud(dataset, index, 3),
        )
    )
