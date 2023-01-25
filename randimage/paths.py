import random
import numpy as np


class BasePath(object):
    def __init__(self, mask) -> None:
        self.mask = mask
        self.mask_shape = mask.shape
        self.max_x, self.max_y = self.mask_shape[0] - 1, self.mask_shape[1] - 1

    def _get_square_neighboorhood(self, point, used_points):
        x, y = point
        cx, cy = self.mask_shape[0] - x, self.mask_shape[1] - y
        max_radius = max(x, cx, y, cy)
        out = set()
        radius = 1

        while not out:
            if radius >= max_radius:
                break

            diff_x = x - radius
            diff_y = y - radius
            sum_x = x + radius
            sum_y = y + radius
            square_neighbors = set()
            for i in range(2 * radius + 1):
                max_x_radius = max(0, min(diff_x + i, self.max_x))
                max_y_radius = max(0, min(diff_y + i, self.max_y))

                square_neighbors.update(
                    (
                        (max(0, diff_x), max_y_radius),
                        (min(self.max_x, sum_x), max_y_radius),
                        (max_x_radius, max(diff_y, 0)),
                        (max_x_radius, min(sum_y, self.max_y)),
                    )
                )

            out = square_neighbors.difference(used_points)
            radius += 1
        return out


class EPWTPath(BasePath):
    def get_path(self):
        x, y = random.randint(0, self.max_x), random.randint(0, self.max_y)
        cur_point = (x, y)
        out = [cur_point]
        used_points = {cur_point}
        while True:
            neighbors = self._get_square_neighboorhood(cur_point, used_points)
            if not neighbors:
                break
            mindiff = np.inf
            for neigh in neighbors:
                diff = np.abs(self.mask[cur_point] - self.mask[neigh])
                if diff < mindiff:
                    mindiff = diff
                    next_point = neigh
            out.append(next_point)
            used_points.add(next_point)
            cur_point = next_point
        return out


class ProbabilisticPath(BasePath):
    def get_path(self, rule=None):
        if rule is None:
            # TODO unused variable
            rule = bin(random.randint(0, 2**12))[2:]
        x, y = random.randint(0, self.max_x), random.randint(0, self.max_y)
        cur_point = (x, y)
        out = [cur_point]
        used_points = {cur_point}
        while True:
            neighbors = self._get_square_neighboorhood(cur_point, used_points)
            if not neighbors:
                break
            neighbors = tuple(neighbors)
            weights = np.zeros(len(neighbors))
            for idx, n in enumerate(neighbors):
                weights[idx] = self.mask[n]
            if np.sum(weights**2) == 0:
                weights[random.randint(0, len(neighbors) - 1)] = 1
            next_point = random.choices(neighbors, weights=weights, k=1).pop()
            out.append(next_point)
            used_points.add(next_point)
            cur_point = next_point
        return out


PATHS = (EPWTPath, ProbabilisticPath)
