import tempfile
import zipfile
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from tqdm import tqdm
from trimesh.base import Trimesh


class ModelProcessor:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def process_all(self):
        for zip_path in tqdm(self.data_dir.glob("*.zip")):
            model_type = zip_path.stem

            with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
                tmp_path = Path(tmp)

                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(tmp_path)

                for obj in tmp_path.rglob("*.obj"):
                    rel_path = obj.relative_to(tmp_path)
                    model_number = rel_path.parts[0]
                    output_dir = self.output_dir / model_type / model_number
                    output_dir.mkdir(parents=True, exist_ok=True)

                    mesh = trimesh.load_mesh(file_obj=obj)
                    normalized_mesh = self.normalize_mesh(mesh)

                    points = self.get_points_from_mesh(normalized_mesh)
                    noise_points1 = self.add_noise_to_points(points, 0.0005)
                    noise_points2 = self.add_noise_to_points(points, 0.0015)
                    noise_points3 = self.add_noise_to_points(points, 0.005)
                    points_with_hole = self.remove_points_hole(points)

                    # self.visualize(points_with_hole)

                    self.save_points(
                        (
                            points,
                            noise_points1,
                            noise_points2,
                            noise_points3,
                            points_with_hole,
                        ),
                        output_dir / "arrays.npz",
                    )
                    break  # Process only one obj file
            break  # Process only one zip file for testing purpouses

    def normalize_mesh(self, mesh: Trimesh) -> Trimesh:
        mesh = mesh.copy()

        # Center around 0,0,0
        mesh.apply_translation(-mesh.bounding_box.centroid)

        # Scale to unit sphere
        max_norm = np.linalg.norm(mesh.vertices, axis=1).max()
        mesh.apply_scale(1.0 / max_norm)

        return mesh

    def get_points_from_mesh(self, mesh: Trimesh, num_of_points=100_000) -> np.ndarray:
        points, _ = trimesh.sample.sample_surface_even(mesh, num_of_points)
        return points

    def add_noise_to_points(self, points: np.ndarray, noise: float) -> np.ndarray:
        return points + np.random.normal(scale=noise, size=points.shape)

    def remove_points_hole(self, points: np.ndarray) -> np.ndarray:
        mask = np.ones(len(points), dtype=bool)

        # Random points from pc
        num_holes = 3
        hole_indices = np.random.choice(len(points), size=num_holes, replace=False)
        hole_centers = points[hole_indices]

        # Make a hole with radius 0.1 at the random points
        radius = 0.1
        for pos in hole_centers:
            mask &= np.linalg.norm(points - pos, axis=1) > radius

        return points[mask]

    def save_points(self, points, file_path: Path | str):
        output_path = Path(file_path)
        np.savez_compressed(
            output_path,
            classic=points[0],
            noise1=points[1],
            noise2=points[2],
            noise3=points[3],
            hole=points[4],
        )

    def visualize(self, points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
        sphere.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(sphere.lines))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        o3d.visualization.draw_geometries([pcd, sphere])


if __name__ == "__main__":
    model_processor = ModelProcessor("Models", "PointClouds")
    model_processor.process_all()
