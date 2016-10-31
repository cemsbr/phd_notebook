"""Get stage statistics."""


class StageStats:
    """Used by parsers to get stage statistics."""

    @staticmethod
    def get_titles(stages):
        """s00_tasks, s00_dur, s01_tasks, s01_dur, ..."""
        return ['s{:02d}_{}'.format(i, stat) for i in range(len(stages))
                for stat in ('tasks', 'in', 'out', 'dur')]

    @staticmethod
    def get_stats(stages):
        """Return number of tasks an duration of each stage."""
        stats = []
        for stage in stages:
            # Use sum to count "sucessful_tasks" generator
            n_tasks = sum(1 for _ in stage.successful_tasks)
            stats.append(n_tasks)
            stats.append(stage.bytes_read)
            stats.append(stage.bytes_written)
            stats.append(stage.duration)
        return stats
