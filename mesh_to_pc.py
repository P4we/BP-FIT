import tempfile
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm
from trimesh.base import Trimesh


class ModelProcessor:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        print("data", self.data_dir.resolve())
        self.output_dir = Path(output_dir)
        print("output", self.output_dir.resolve())
        self.output_dir.mkdir(exist_ok=True)

    def process_all(self):
        for zip_path in tqdm(self.data_dir.glob("*.zip")):
            print("zip path", zip_path.resolve())
            model_type = zip_path.stem
            with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
                tmp_path = Path(tmp)
                print("tmp path", tmp_path.resolve())
                with zipfile.ZipFile(zip_path) as zip:
                    zip.extractall(tmp_path)

                for obj in tmp_path.rglob("*.obj"):
                    print("processing obj: ", obj)

                    rel_path = obj.relative_to(tmp_path)
                    model_number = rel_path.parts[0]
                    output_dir = self.output_dir / model_type / model_number
                    output_dir.mkdir(parents=True, exist_ok=True)

                    mesh = trimesh.load_mesh(file_obj=obj)
                    print(mesh)
                    normalized_mesh = self.normalize_mesh(mesh)
                    print(normalized_mesh)
                    points = self.get_points_from_mesh(normalized_mesh)
                    print(points)
                    noise_points1 = self.add_noise_to_points(points, 0.0005)
                    noise_points2 = self.add_noise_to_points(points, 0.0015)
                    noise_points3 = self.add_noise_to_points(points, 0.005)
                    points_with_hole = self.remove_points_hole(points)
                    print(points_with_hole)
                    self.save_points(points, output_dir / "classic.npy")
                    self.save_points(noise_points1, output_dir / "noise1.npy")
                    self.save_points(noise_points2, output_dir / "noise2.npy")
                    self.save_points(noise_points3, output_dir / "noise3.npy")
                    self.save_points(points_with_hole, output_dir / "hole.npy")
                    break  # Process only one obj file
            break  # Process only one zip file for testing purpouses

    def normalize_mesh(self, mesh: Trimesh) -> Trimesh:
        mesh.apply_translation(  # Center the mesh around 0,0,0
            -mesh.bounding_box.centroid
        )
        scale = 1.0 / mesh.scale  # Scaling to a box (-1, 1)
        mesh.apply_scale(scale)
        return mesh

    def get_points_from_mesh(self, mesh: Trimesh, num_of_points=100_000) -> np.ndarray:
        points, _ = trimesh.sample.sample_surface_even(mesh, num_of_points)
        return points

    def add_noise_to_points(self, points: np.ndarray, noise: float) -> np.ndarray:
        return points + np.random.normal(scale=noise, size=points.shape)

    def remove_points_hole(self, points: np.ndarray) -> np.ndarray:
        pos = np.random.uniform(-1, 1, size=3)
        radius = 0.05
        mask = np.linalg.norm(points - pos, axis=1) > radius
        return points[mask]

    def save_points(self, points: np.ndarray, file_path: Path | str):
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, points)


if __name__ == "__main__":
    model_processor = ModelProcessor("Models", "data_output")
    model_processor.process_all()
