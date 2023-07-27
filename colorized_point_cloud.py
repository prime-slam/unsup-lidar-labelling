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

import open3d as o3d
import numpy as np


def get_list_of_point_colors_from_image(points, image):
    pixels = list(image.getdata())
    colors = []
    for i in range(points.shape[1]):
        colors.append(pixels[(int(points[1, i]) * image.width + int(points[0, i]))])
    for i in range(len(colors)):
        colors[i] = (colors[i][0] / 255, colors[i][1] / 255, colors[i][2] / 255)
    return colors


def __delete_point_which_not_visible_on_image(points, image):
    points = points[:, points[0] >= 0]
    points = points[:, points[1] >= 0]
    points = points[:, points[2] >= 0]
    points_depth = points[2]
    points = points[:2, :] / points[2, :]
    points = np.vstack((points, points_depth))
    points = points[:, points[0] <= image.width]
    points = points[:, points[1] <= image.height]
    return points


def get_colorized_2d_point_cloud(dataset, index_of_image, cam_number):
    velo_data = dataset.get_velo(index_of_image)
    if cam_number == 2:
        image = dataset.get_cam2(index_of_image)
    else:
        image = dataset.get_cam3(index_of_image)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(velo_data[:, :3])
    if cam_number == 2:
        point_cloud.transform(dataset.calib.T_cam2_velo)
        points3d_coordinates_in_2d = (
            dataset.calib.K_cam2 @ np.asarray(point_cloud.points).transpose()
        )
    else:
        point_cloud.transform(dataset.calib.T_cam3_velo)
        points3d_coordinates_in_2d = (
            dataset.calib.K_cam3 @ np.asarray(point_cloud.points).transpose()
        )
    points3d_coordinates_in_2d = __delete_point_which_not_visible_on_image(
        points3d_coordinates_in_2d, image
    )
    point_cloud.points = o3d.utility.Vector3dVector(
        points3d_coordinates_in_2d.transpose()
    )
    point_cloud.colors = o3d.utility.Vector3dVector(
        get_list_of_point_colors_from_image(points3d_coordinates_in_2d, image)
    )
    return point_cloud


def get_colorized_3d_point_cloud(dataset, index_of_image, cam_number):
    velo_data = dataset.get_velo(index_of_image)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(velo_data[:, :3])
    if cam_number == 2:
        image = dataset.get_cam2(index_of_image)
        point_cloud.transform(dataset.calib.T_cam2_velo)
        points3d_coordinates_in_2d = (
            dataset.calib.K_cam2 @ np.asarray(point_cloud.points).transpose()
        )
    else:
        image = dataset.get_cam3(index_of_image)
        point_cloud.transform(dataset.calib.T_cam3_velo)
        points3d_coordinates_in_2d = (
            dataset.calib.K_cam3 @ np.asarray(point_cloud.points).transpose()
        )
    points3d_coordinates_in_2d = __delete_point_which_not_visible_on_image(
        points3d_coordinates_in_2d, image
    )
    points_depth = points3d_coordinates_in_2d[2]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.colors = o3d.utility.Vector3dVector(
        get_list_of_point_colors_from_image(points3d_coordinates_in_2d, image)
    )
    points3d_coordinates_in_2d = (
        points3d_coordinates_in_2d[:2, :] * points3d_coordinates_in_2d[2, :]
    )
    points3d_coordinates_in_2d = np.vstack((points3d_coordinates_in_2d, points_depth))
    if cam_number == 2:
        new_points = np.linalg.inv(dataset.calib.K_cam2) @ points3d_coordinates_in_2d
    else:
        new_points = np.linalg.inv(dataset.calib.K_cam3) @ points3d_coordinates_in_2d
    point_cloud.points = o3d.utility.Vector3dVector(new_points.transpose())
    return point_cloud


def get_cloud_union_in_world_coords(dataset, cloud_2, cloud_3):
    cloud_2.transform(np.linalg.inv(dataset.calib.T_cam2_velo)).transform(
        dataset.calib.T_cam0_velo
    ).transform(dataset.poses[0])
    cloud_3.transform(np.linalg.inv(dataset.calib.T_cam3_velo)).transform(
        dataset.calib.T_cam0_velo
    ).transform(dataset.poses[0])
    cloud_3 += cloud_2
    return cloud_3
