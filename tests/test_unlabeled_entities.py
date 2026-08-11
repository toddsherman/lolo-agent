from __future__ import annotations

import unittest

from lolo_agent.environment import Action
from lolo_agent.pixels import Frame
from lolo_agent.unlabeled_entities import UnlabeledEntityMemory


def grid_frame(cells: list[list[tuple[int, int, int]]]) -> Frame:
    rows = len(cells)
    columns = len(cells[0])
    tile = 4
    pixels = bytearray(columns * tile * rows * tile * 3)
    for row, values in enumerate(cells):
        for column, color in enumerate(values):
            for y in range(row * tile, (row + 1) * tile):
                for x in range(column * tile, (column + 1) * tile):
                    offset = (y * columns * tile + x) * 3
                    pixels[offset : offset + 3] = bytes(color)
    return Frame(columns * tile, rows * tile, 3, bytes(pixels))


class UnlabeledEntityMemoryTests(unittest.TestCase):
    def test_recurring_patch_is_assigned_to_same_prototype(self) -> None:
        memory = UnlabeledEntityMemory(
            columns=3,
            rows=2,
            pooled_columns=2,
            pooled_rows=2,
            match_threshold=0.05,
        )
        floor = (16, 16, 16)
        entity = (240, 32, 160)
        first = memory.observe(
            grid_frame([[floor, entity, floor], [floor, floor, floor]])
        )
        second = memory.observe(
            grid_frame([[floor, floor, floor], [floor, entity, floor]])
        )
        self.assertEqual(first.prototype_at(1, 0), second.prototype_at(1, 1))
        self.assertNotEqual(first.prototype_at(0, 0), first.prototype_at(1, 0))
        entity_id = first.prototype_at(1, 0)
        entity_stats = memory.stats()[entity_id]
        self.assertEqual(entity_stats.unique_cells, 2)
        self.assertEqual(entity_stats.frames_observed, 2)

    def test_small_appearance_variation_uses_existing_prototype(self) -> None:
        memory = UnlabeledEntityMemory(
            columns=1,
            rows=1,
            pooled_columns=2,
            pooled_rows=2,
            quantization=16,
            match_threshold=0.1,
        )
        first = memory.observe(grid_frame([[(128, 64, 32)]]))
        second = memory.observe(grid_frame([[(136, 72, 32)]]))
        self.assertEqual(first.prototype_at(0, 0), second.prototype_at(0, 0))
        self.assertEqual(memory.prototype_count, 1)

    def test_masked_feature_ignores_overlapping_sprite_pixels(self) -> None:
        memory = UnlabeledEntityMemory(
            columns=1,
            rows=1,
            pooled_columns=2,
            pooled_rows=2,
            quantization=16,
        )
        first_pixels = bytearray(4 * 4 * 3)
        second_pixels = bytearray(4 * 4 * 3)
        ignored = set()
        for y in range(4):
            for x in range(4):
                offset = (y * 4 + x) * 3
                if x < 2:
                    first_pixels[offset : offset + 3] = bytes((255, 255, 255))
                    second_pixels[offset : offset + 3] = bytes((0, 0, 255))
                    ignored.add((x, y))
                else:
                    first_pixels[offset : offset + 3] = bytes((224, 32, 160))
                    second_pixels[offset : offset + 3] = bytes((224, 32, 160))
        first = Frame(4, 4, 3, bytes(first_pixels))
        second = Frame(4, 4, 3, bytes(second_pixels))

        self.assertNotEqual(
            memory.feature_at(first, 0, 0),
            memory.feature_at(second, 0, 0),
        )
        self.assertEqual(
            memory.feature_at(first, 0, 0, ignored),
            memory.feature_at(second, 0, 0, ignored),
        )

    def test_directional_interactions_are_unlabeled_coverage_edges(self) -> None:
        memory = UnlabeledEntityMemory(
            columns=3,
            rows=3,
            pooled_columns=2,
            pooled_rows=2,
        )
        floor = (16, 16, 16)
        entity = (240, 32, 160)
        observation = memory.observe(
            grid_frame(
                [
                    [floor, floor, floor],
                    [floor, floor, entity],
                    [floor, floor, floor],
                ]
            )
        )
        target = memory.target_prototype(
            observation,
            anchor=(4, 4),
            action=Action.RIGHT,
            frame_width=12,
            frame_height=12,
        )
        self.assertEqual(target, observation.prototype_at(2, 1))
        self.assertEqual(
            memory.action_target_cell(
                anchor=(4, 4),
                action=Action.RIGHT,
                frame_width=12,
                frame_height=12,
            ),
            (2, 1),
        )
        self.assertEqual(memory.record_interaction(target, Action.RIGHT, 16), 0)
        self.assertEqual(memory.record_interaction(target, Action.RIGHT, 16), 1)
        self.assertEqual(memory.record_interaction(target, Action.LEFT, 16), 0)

    def test_non_directional_action_has_no_target_patch(self) -> None:
        memory = UnlabeledEntityMemory(columns=1, rows=1)
        observation = memory.observe(grid_frame([[(16, 16, 16)]]))
        self.assertIsNone(
            memory.target_prototype(
                observation,
                anchor=(0, 0),
                action=Action.A,
                frame_width=4,
                frame_height=4,
            )
        )


if __name__ == "__main__":
    unittest.main()
