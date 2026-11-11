import unittest

from src.perception.analyzeCollisionRisk import MotionHistory, cross_conflicts, predict_conflict


class CollisionRiskTests(unittest.TestCase):
    def test_approaching_vehicle_enters_radius(self):
        prediction = predict_conflict(((0, 0), (0, 0)), ((10, 0), (-2, 0)), 2, 5)
        self.assertAlmostEqual(prediction[0], 4)
        self.assertAlmostEqual(prediction[1], 0)

    def test_crossing_trajectories(self):
        prediction = predict_conflict(((0, -5), (0, 1)), ((-5, 0), (1, 0)), 1, 6)
        self.assertIsNotNone(prediction)
        self.assertAlmostEqual(prediction[1], 0)

    def test_separating_parallel_and_late_objects(self):
        pedestrian = ((0, 0), (0, 0))
        for vehicle in [((10, 0), (2, 0)), ((10, 0), (0, 0)),
                        ((10, 5), (-2, 0)), ((100, 0), (-2, 0))]:
            with self.subTest(vehicle=vehicle):
                self.assertIsNone(predict_conflict(pedestrian, vehicle, 2, 5))

    def test_current_proximity_and_missing_motion(self):
        self.assertEqual(predict_conflict(((0, 0), (0, 0)), ((1, 0), (0, 0)), 2, 3), (0, 1))
        self.assertIsNone(predict_conflict(((0, 0), None), ((1, 0), (0, 0)), 2, 3))

    def test_only_vehicle_pedestrian_pairs_even_with_equal_ids(self):
        def obj(kind):
            return dict(kind=kind, id=1, point=(0, 0), velocity=(0, 0))
        for kind in ('veiculo', 'pedestre'):
            self.assertEqual(list(cross_conflicts([obj(kind), obj(kind)], 2, 3)), [])
        self.assertEqual(len(list(cross_conflicts([obj('veiculo'), obj('pedestre')], 2, 3))), 1)

    def test_speed_uses_video_time_and_resets_after_gap(self):
        history = MotionHistory()
        self.assertIsNone(history.update(('pedestre', 1), 0, (0, 0)))
        for index in range(1, 7):
            velocity = history.update(('pedestre', 1), index / 10, (index / 5, 0))
        self.assertAlmostEqual(velocity[0], 2)
        self.assertAlmostEqual(velocity[1], 0)
        self.assertIsNone(history.update(('pedestre', 1), 1.5, (30, 0)))
        history.expire(2)
        self.assertEqual(history.samples, {})


if __name__ == '__main__':
    unittest.main()
