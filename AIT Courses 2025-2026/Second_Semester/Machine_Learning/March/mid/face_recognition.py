from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 42
RESIZE_TO = (96, 96)


@dataclass
class LabeledImage:
	path: Path
	image: np.ndarray
	label: int


def augment_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
	h, w = image.shape[:2]
	angle = float(rng.uniform(-8.0, 8.0))
	scale = float(rng.uniform(0.96, 1.04))
	tx = float(rng.uniform(-0.03 * w, 0.03 * w))
	ty = float(rng.uniform(-0.03 * h, 0.03 * h))

	mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
	mat[:, 2] += [tx, ty]

	out = cv2.warpAffine(
		image,
		mat,
		(w, h),
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_REFLECT,
	)

	alpha = float(rng.uniform(0.9, 1.1))
	beta = float(rng.uniform(-8.0, 8.0))
	out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
	return out


def expand_train_set(images: Sequence[np.ndarray], labels: np.ndarray, copies_per_image: int = 25):
	rng = np.random.default_rng(SEED)
	x_images: List[np.ndarray] = []
	y_labels: List[int] = []

	for image, label in zip(images, labels):
		x_images.append(image)
		y_labels.append(int(label))
		for _ in range(copies_per_image):
			x_images.append(augment_image(image, rng))
			y_labels.append(int(label))

	return x_images, np.array(y_labels, dtype=np.int32)


def load_images(folder: Path) -> List[Tuple[Path, np.ndarray]]:
	extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
	items: List[Tuple[Path, np.ndarray]] = []
	for path in sorted(folder.iterdir()):
		if path.suffix.lower() not in extensions:
			continue
		image = cv2.imread(str(path))
		if image is None:
			continue
		items.append((path, image))
	return items


def to_gray_resized(image: np.ndarray, size: Tuple[int, int] = RESIZE_TO) -> np.ndarray:
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
	return gray


def map_changed_to_base(
	base_images: Sequence[Tuple[Path, np.ndarray]],
	changed_images: Sequence[Tuple[Path, np.ndarray]],
) -> Dict[Path, int]:
	# Auto-label changed images by nearest base image in normalized pixel space.
	base_vectors = [to_gray_resized(img).astype(np.float32) / 255.0 for _, img in base_images]
	mapping: Dict[Path, int] = {}

	for changed_path, changed_img in changed_images:
		changed_vec = to_gray_resized(changed_img).astype(np.float32) / 255.0
		distances = [np.mean((changed_vec - base_vec) ** 2) for base_vec in base_vectors]
		best_idx = int(np.argmin(distances))
		mapping[changed_path] = best_idx
	return mapping


def detect_face_box(image: np.ndarray, face_cascade: cv2.CascadeClassifier) -> Tuple[int, int, int, int]:
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

	if len(faces) == 0:
		h, w = gray.shape
		return int(0.15 * w), int(0.1 * h), int(0.7 * w), int(0.8 * h)

	x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
	return int(x), int(y), int(w), int(h)


def extract_face_region(image: np.ndarray, face_cascade: cv2.CascadeClassifier) -> np.ndarray:
	x, y, w, h = detect_face_box(image, face_cascade)

	pad_w = int(0.08 * w)
	pad_h = int(0.08 * h)
	x1 = max(0, x - pad_w)
	y1 = max(0, y - pad_h)
	x2 = min(image.shape[1], x + w + pad_w)
	y2 = min(image.shape[0], y + h + pad_h)
	return image[y1:y2, x1:x2]


def mask_rect(image: np.ndarray, rect: Tuple[int, int, int, int], degree: float) -> None:
	x, y, w, h = rect
	if degree <= 0:
		return

	cx = x + w / 2.0
	cy = y + h / 2.0
	nw = max(1, int(w * degree))
	nh = max(1, int(h * degree))
	x1 = max(0, int(cx - nw / 2.0))
	y1 = max(0, int(cy - nh / 2.0))
	x2 = min(image.shape[1], int(cx + nw / 2.0))
	y2 = min(image.shape[0], int(cy + nh / 2.0))
	image[y1:y2, x1:x2] = 0


def ablate_feature(
	image: np.ndarray,
	feature: str,
	degree: float,
	face_cascade: cv2.CascadeClassifier,
	eye_cascade: cv2.CascadeClassifier,
) -> np.ndarray:
	out = image.copy()
	x, y, w, h = detect_face_box(out, face_cascade)
	face_roi = out[y : y + h, x : x + w]

	if feature == "eyes":
		gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
		eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(12, 12))

		# Keep eye candidates from the upper half of the face.
		eyes = [e for e in eyes if e[1] + e[3] < int(0.65 * h)]
		eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

		if len(eyes) == 0:
			approx = [
				(int(x + 0.2 * w), int(y + 0.28 * h), int(0.22 * w), int(0.12 * h)),
				(int(x + 0.58 * w), int(y + 0.28 * h), int(0.22 * w), int(0.12 * h)),
			]
			for rect in approx:
				mask_rect(out, rect, degree)
		else:
			for ex, ey, ew, eh in eyes:
				rect = (x + ex, y + ey, ew, eh)
				mask_rect(out, rect, degree)

	elif feature == "nose":
		# Nose estimated as central-lower part of face for robust behavior without landmarks.
		nx = int(x + 0.34 * w)
		ny = int(y + 0.44 * h)
		nw = int(0.32 * w)
		nh = int(0.28 * h)
		mask_rect(out, (nx, ny, nw, nh), degree)

	return out


def build_feature_matrix(images: Sequence[np.ndarray], face_cascade: cv2.CascadeClassifier) -> np.ndarray:
	vectors: List[np.ndarray] = []
	for image in images:
		face = extract_face_region(image, face_cascade)
		gray = to_gray_resized(face)
		eq = cv2.equalizeHist(gray)
		vectors.append(eq.flatten().astype(np.float32) / 255.0)
	return np.vstack(vectors)


def train_models(x_train: np.ndarray, y_train: np.ndarray):
	svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=4.0, gamma="scale", random_state=SEED))
	rf = RandomForestClassifier(n_estimators=300, random_state=SEED)

	svm.fit(x_train, y_train)
	rf.fit(x_train, y_train)
	return svm, rf


def evaluate_condition(
	x_train: np.ndarray,
	y_train: np.ndarray,
	test_images: Sequence[np.ndarray],
	y_test: np.ndarray,
	feature: str | None,
	degree: float,
	face_cascade: cv2.CascadeClassifier,
	eye_cascade: cv2.CascadeClassifier,
) -> Dict[str, float]:
	if feature is None:
		eval_images = list(test_images)
		condition_name = "baseline"
	else:
		eval_images = [
			ablate_feature(img, feature=feature, degree=degree, face_cascade=face_cascade, eye_cascade=eye_cascade)
			for img in test_images
		]
		condition_name = f"{feature}_{int(degree * 100)}"

	x_test = build_feature_matrix(eval_images, face_cascade)
	svm, rf = train_models(x_train, y_train)
	svm_acc = accuracy_score(y_test, svm.predict(x_test))
	rf_acc = accuracy_score(y_test, rf.predict(x_test))

	return {
		"condition": condition_name,
		"svm_accuracy": float(svm_acc),
		"rf_accuracy": float(rf_acc),
	}


def prepare_dataset(base_dir: Path, changed_dir: Path) -> Tuple[List[LabeledImage], List[LabeledImage]]:
	base_images = load_images(base_dir)
	changed_images = load_images(changed_dir)

	if len(base_images) < 2:
		raise ValueError("Need at least 2 base images in my_faces folder.")
	if len(changed_images) == 0:
		raise ValueError("Need at least 1 changed image in my_changed_faces folder.")

	base_labeled: List[LabeledImage] = []
	for i, (path, image) in enumerate(base_images):
		base_labeled.append(LabeledImage(path=path, image=image, label=i))

	mapping = map_changed_to_base(base_images, changed_images)
	changed_labeled: List[LabeledImage] = []
	for path, image in changed_images:
		changed_labeled.append(LabeledImage(path=path, image=image, label=mapping[path]))

	return base_labeled, changed_labeled


def print_results(results: Sequence[Dict[str, float]]) -> None:
	print("\n=== Accuracy Comparison (same faces, same ablation, two models) ===")
	print(f"{'Condition':<15} {'SVM':>10} {'RandomForest':>14}")
	print("-" * 42)
	for row in results:
		print(f"{row['condition']:<15} {row['svm_accuracy']*100:>9.2f}% {row['rf_accuracy']*100:>13.2f}%")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Face recognition ablation study on eyes and nose using SVM and RandomForest"
	)
	parser.add_argument("--base-dir", type=str, default="my_faces", help="Folder with original base faces")
	parser.add_argument(
		"--changed-dir", type=str, default="my_changed_faces", help="Folder with modified/changed faces"
	)
	parser.add_argument(
		"--degrees",
		type=float,
		nargs="*",
		default=[0.25, 0.5, 0.75, 1.0],
		help="Ablation degrees to test. Use 1.0 for 100%% feature removal.",
	)
	args = parser.parse_args()

	script_dir = Path(__file__).resolve().parent
	base_dir = (script_dir / args.base_dir).resolve()
	changed_dir = (script_dir / args.changed_dir).resolve()

	base_data, changed_data = prepare_dataset(base_dir, changed_dir)

	train_images_raw = [item.image for item in base_data]
	y_train_raw = np.array([item.label for item in base_data], dtype=np.int32)

	test_images = [item.image for item in changed_data]
	y_test = np.array([item.label for item in changed_data], dtype=np.int32)

	if len(np.unique(y_train_raw)) < 2:
		raise ValueError("Need at least 2 identities/classes in my_faces for meaningful accuracy comparison.")

	train_images, y_train = expand_train_set(train_images_raw, y_train_raw, copies_per_image=30)
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

	if face_cascade.empty() or eye_cascade.empty():
		raise RuntimeError("OpenCV Haar cascades not found. Check OpenCV installation.")

	x_train = build_feature_matrix(train_images, face_cascade)

	results: List[Dict[str, float]] = []

	# Baseline without ablation.
	results.append(
		evaluate_condition(
			x_train,
			y_train,
			test_images,
			y_test,
			feature=None,
			degree=0.0,
			face_cascade=face_cascade,
			eye_cascade=eye_cascade,
		)
	)

	for feature in ("eyes", "nose"):
		for degree in args.degrees:
			results.append(
				evaluate_condition(
					x_train,
					y_train,
					test_images,
					y_test,
					feature=feature,
					degree=float(degree),
					face_cascade=face_cascade,
					eye_cascade=eye_cascade,
				)
			)

	print_results(results)


if __name__ == "__main__":
	main()
