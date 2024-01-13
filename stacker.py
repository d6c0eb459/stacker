"""
A utility for putting things on top of other things.

Features:
    - "Drop down" mode stacks objects on top of others that are directly below it, as
      determined by simple bounding box collision.
    - "Stack above" mode moves objects into a centered single file stack, optionally
      sorting first.

Standards:
    - black for autoformatting.
    - pylint for static tests.
    - unittest for unit tests.
    - Google's style guide for doc strings.
"""
import unittest

import numpy as np


def get_base(bounds, axis):
    """
    Returns the bound with the lowest start value in the given axis.

    Arguments:
        bounds: A np.array of shape (num_bounds, 3, 2).
        axis:   0, 1, or 2 for x, y, z.
    """
    idx = np.argmin(bounds[:, axis, 0])
    return bounds[idx]


def argsort_by_area(bounds, axis):
    """
    Returns the given bounds sorted by area, smallest first.

    The area is determined by treating the given axis as having zero length.

    Arguments:
        bounds: A np.array of shape (num_bounds, 3, 2).
        axis:   0, 1, or 2 for x, y, z.
    """
    dimensions = bounds[:, :, 1] - bounds[:, :, 0]
    volume = np.prod(dimensions, 1)
    area = volume / dimensions[:, axis]

    return np.argsort(area)


def argsort_by_height(bounds, axis):
    """
    Returns the given bounds sorted by height, lowest first.

    Arguments:
        bounds: A np.array of shape (num_bounds, 3, 2).
        axis:   0, 1, or 2 for x, y, z.
    """
    return np.argsort(bounds[:, axis, 0])


def stack_above(base_bounds, incoming_bounds, axis, padding=0, centering=True):
    """
    Returns the amount that the incoming bounds should be translated by in order to
    be stacked on top of the given base bounds.

    Arguments:
        base_bounds:     A np.array of shape (3, 2).
        incoming_bounds: A np.array of shape (3, 2).
        axis:            0, 1, or 2 for x, y, z.
        padding:         (optional) Extra space to put between each object.
        centering:       (optional) Defaults to centering the incoming bounds over the
                         given base; pass False to disable.
    """
    if centering:
        # Use the center of each bound as a reference
        source = np.mean(incoming_bounds, axis=-1)
        target = np.mean(base_bounds, axis=-1)
    else:
        # Just use incoming's original position
        source = incoming_bounds[:, 0].copy()
        target = source.copy()

    # Bottom of incoming should be at top of base
    source[axis] = incoming_bounds[axis][0]  # bottom
    target[axis] = base_bounds[axis][1]  # top

    target[axis] += padding

    return target - source


def is_below(bounds_a, bounds_b, axis):
    """
    Returns True if the first given bound is anywhere below (ie in the shadow of) the
    second given bound, and False otherwise.

    Arguments:
        bounds_a: A np.array of shape (3, 2).
        bounds_b: A np.array of shape (3, 2).
        axis:     0, 1, or 2 for x, y, z.
    """
    # simple bounding box collision for other axes
    collision = (bounds_b[:, 0] < bounds_a[:, 1]) & (bounds_b[:, 1] > bounds_a[:, 0])
    collision |= (bounds_b[:, 1] > bounds_a[:, 0]) & (bounds_b[:, 0] < bounds_a[:, 1])

    # set given axis to True if anywhere below or colliding
    collision[axis] = bounds_b[axis, 1] >= bounds_a[axis, 0]

    return all(collision)


def drop_down(bounds, axis, padding=0):
    """
    Returns the amount that each of the given bounds should be translated in order to
    be stacked on top of other bounds appearing directly below them.

    Arguments:
        bounds:  A np.array of shape (num_bounds, 3, 2).
        axis:    0, 1, or 2 for x, y, z.
        padding: (optional) Extra space to put between each object.
    """
    bounds = bounds.copy()

    idxs = argsort_by_height(bounds, axis)

    floor = bounds[idxs[0]].copy()
    floor[:, 1] = floor[:, 0]  # make flat

    output = np.zeros((len(bounds), 3))
    for idx in range(1, len(idxs)):
        i = idxs[idx]

        source = bounds[i]

        candidates = [bounds[j] for j in idxs[:idx]]
        candidates = [b for b in candidates if is_below(b, source, axis)]
        candidates = np.asarray([floor] + list(candidates))

        target = candidates[np.argmax(candidates[:, axis, 1])]

        output[i] = stack_above(target, source, axis, centering=False)
        if not np.array_equal(target, floor):
            output[i, axis] += padding

        bounds[i] = translate_bounds(bounds[i], output[i])

    return output


def translate_bounds(bounds, delta):
    """
    Returns new bounds translated by the given delta.

    Arguments:
        bounds: A np.array of shape (3, 2).
        delta:  A np.array of shape (3,).
    """
    return bounds + np.tile(delta, (2, 1)).T


class TestStacker(unittest.TestCase):
    """
    Tests for this module.
    """

    def test_get_base(self):
        """
        Test getting the base.
        """
        bounds = np.asarray(
            [
                [[1, 3], [1, 3], [1, 3]],
                [[1, 3], [3, 4], [2, 3]],
                [[1, 2], [3, 4], [3, 4]],
                [[1, 2], [3, 7], [0, 4]],
            ]
        )

        actual = get_base(bounds, 2)
        expected = [[1, 2], [3, 7], [0, 4]]

        np.testing.assert_equal(actual, expected)

    def test_stack_above(self):
        """
        Test that objects are stacked correctly.
        """
        base_bounds = np.asarray([[1, 2], [3, 4], [5, 6]])
        incoming_bounds = np.asarray([[10, 11], [12, 13], [14, 15]])

        actual = stack_above(base_bounds, incoming_bounds, 0)
        expected = np.asarray([-8, -9, -9])
        np.testing.assert_equal(actual, expected)

        actual = stack_above(base_bounds, incoming_bounds, 1)
        expected = np.asarray([-9, -8, -9])
        np.testing.assert_equal(actual, expected)

        actual = stack_above(base_bounds, incoming_bounds, 2)
        expected = np.asarray([-9, -9, -8])
        np.testing.assert_equal(actual, expected)

    def test_stack_above_no_centering(self):
        """
        Test that objects are stacked correctly when centering is disabled.
        """
        base_bounds = np.asarray([[1, 2], [3, 4], [5, 6]])
        incoming_bounds = np.asarray([[10, 11], [12, 13], [14, 15]])

        actual = stack_above(base_bounds, incoming_bounds, 0, centering=False)
        expected = np.asarray([-8, 0, 0])
        np.testing.assert_equal(actual, expected)

        actual = stack_above(base_bounds, incoming_bounds, 1, centering=False)
        expected = np.asarray([0, -8, 0])
        np.testing.assert_equal(actual, expected)

        actual = stack_above(base_bounds, incoming_bounds, 2, centering=False)
        expected = np.asarray([0, 0, -8])
        np.testing.assert_equal(actual, expected)

    def test_stack_above_padding(self):
        """
        Test that objects are stacked with padding.
        """
        base_bounds = np.asarray([[1, 2], [3, 4], [5, 6]])
        incoming_bounds = np.asarray([[10, 11], [12, 13], [14, 15]])

        actual = stack_above(base_bounds, incoming_bounds, 2, padding=0)
        expected = np.asarray([-9, -9, -8])
        np.testing.assert_equal(actual, expected)

        actual = stack_above(base_bounds, incoming_bounds, 2, padding=1)
        expected = np.asarray([-9, -9, -7])
        np.testing.assert_equal(actual, expected)

    def test_translate_bounds(self):
        """
        Test that bounds are translated correctly.
        """
        bounds = np.asarray([[1, 2], [3, 4], [5, 6]])
        delta = np.asarray([7, 8, 9])

        actual = translate_bounds(bounds, delta)
        expected = np.asarray([[8, 9], [11, 12], [14, 15]])

        np.testing.assert_equal(actual, expected)

    def test_sort_by_area(self):
        """
        Test sorting by area.
        """
        bounds = np.asarray(
            [
                [[1, 3], [1, 3], [1, 3]],
                [[1, 3], [3, 4], [1, 3]],
                [[1, 2], [3, 4], [1, 2]],
                [[1, 2], [3, 7], [1, 2]],
            ]
        )

        actual = argsort_by_area(bounds, 2)
        expected = [2, 1, 0, 3]

        np.testing.assert_equal(actual, expected)

    def test_sort_by_height(self):
        """
        Test sorting by height.
        """
        bounds = np.asarray(
            [
                [[1, 3], [1, 3], [1, 3]],
                [[1, 3], [3, 4], [0, 3]],
                [[1, 2], [3, 4], [3, 2]],
                [[1, 2], [3, 7], [2, 2]],
            ]
        )

        actual = argsort_by_height(bounds, 2)
        expected = [1, 0, 3, 2]

        np.testing.assert_equal(actual, expected)

    def test_is_below(self):
        """
        Test shadow detection.
        """
        bounds = np.asarray(
            [
                [[1, 3], [1, 3], [1, 3]],
                [[2, 3], [2, 4], [2, 3]],
                [[2, 3], [2, 4], [4, 5]],
                [[0, 1], [1, 3], [1, 3]],
                [[3, 4], [2, 4], [1, 3]],
                [[1, 3], [0, 1], [1, 3]],
                [[1, 3], [3, 4], [1, 3]],
                [[8, 9], [8, 9], [1, 3]],
            ]
        )

        # Intersects
        self.assertTrue(is_below(bounds[1], bounds[0], 2))
        self.assertTrue(is_below(bounds[0], bounds[1], 2))

        # Directly above/below
        self.assertTrue(is_below(bounds[1], bounds[2], 2))
        self.assertFalse(is_below(bounds[2], bounds[1], 2))

        # Too far to the left/right
        self.assertFalse(is_below(bounds[0], bounds[3], 2))
        self.assertFalse(is_below(bounds[0], bounds[4], 2))

        # Too far forwards/back
        self.assertFalse(is_below(bounds[0], bounds[5], 2))
        self.assertFalse(is_below(bounds[0], bounds[6], 2))

        # Too far both
        self.assertFalse(is_below(bounds[0], bounds[7], 2))

    def test_drop_down_above(self):
        """
        Test that objects above others are stacked in columns correctly.
        """
        bounds = np.asarray(
            [
                [[1, 3], [1, 3], [1, 2]],
                [[2, 3], [2, 4], [3, 4]],
                [[0, 2], [0, 2], [5, 6]],
            ]
        )

        actual = drop_down(bounds, 2)
        expected = np.asarray(
            [
                [0, 0, 0],
                [0, 0, -1],
                [0, 0, -3],
            ]
        )

        np.testing.assert_equal(actual, expected)

    def test_drop_down_not_above(self):
        """
        Test that objects not above others are stacked in columns correctly.
        """
        bounds = np.asarray(
            [
                [[1, 2], [1, 2], [1, 2]],
                [[2, 3], [2, 3], [3, 4]],
                [[2, 3], [2, 3], [4, 5]],
            ]
        )

        actual = drop_down(bounds, 2)
        expected = np.asarray(
            [
                [0, 0, 0],
                [0, 0, -2],
                [0, 0, -2],
            ]
        )

        np.testing.assert_equal(actual, expected)

    def test_drop_down_out_of_order(self):
        """
        Test that drop down ignores non overlapping even when out of order.
        """
        # base, no overlap, overlap
        bounds = np.asarray(
            [
                [[0, 1], [0, 1], [0, 1]],
                [[0, 1], [1, 2], [4, 5]],
                [[0, 1], [0, 1], [2, 3]],
            ]
        )

        actual = drop_down(bounds, 2)
        expected = np.asarray(
            [
                [0, 0, 0],
                [0, 0, -4],
                [0, 0, -1],
            ]
        )

        np.testing.assert_equal(actual, expected)

    def test_drop_down_unstable_order(self):
        """
        Test that drop down works even when moving objects underneath changes their
        order.
        """
        # base, no overlap, overlap both
        bounds = np.asarray(
            [
                [[0, 1], [0, 1], [0, 4]],
                [[0, 1], [2, 3], [2, 3]],
                [[0, 1], [0, 1], [1, 5]],
            ]
        )

        actual = drop_down(bounds, 2)
        expected = np.asarray(
            [
                [0, 0, 0],
                [0, 0, -2],
                [0, 0, 3],
            ]
        )

        np.testing.assert_equal(actual, expected)

    def test_drop_down_no_floor_padding(self):
        """
        Test that drop down does not add padding to the floor.
        """
        # base, no overlap, overlap both
        bounds = np.asarray(
            [
                [[0, 1], [0, 1], [0, 4]],
                [[0, 1], [2, 3], [2, 3]],
            ]
        )

        actual = drop_down(bounds, 2, padding=1)
        expected = np.asarray(
            [
                [0, 0, 0],
                [0, 0, -2],
            ]
        )

        np.testing.assert_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
