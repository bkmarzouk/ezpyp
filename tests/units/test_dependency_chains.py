from pathlib import Path

from ezpyp.steps import PickleStep, expand_dependencies, reduce_dependencies


class TestDependencyExpansion:
    @staticmethod
    def test_empty_deps():
        assert expand_dependencies([]) == []

    @staticmethod
    def test_no_deps():
        s1 = PickleStep(Path.cwd(), "test", [], {}, lambda x: None, [])
        assert expand_dependencies([s1]) == [s1]

    @staticmethod
    def test_linear_deps():
        s1 = PickleStep(Path.cwd(), "s1", [], {}, lambda x: None, [])
        s2 = PickleStep(Path.cwd(), "s2", [], {}, lambda x: None, [s1])
        assert expand_dependencies([s2]) == [s2, s1]

        s3 = PickleStep(Path.cwd(), "s3", [], {}, lambda x: None, [s2])
        assert expand_dependencies([s3]) == [s3, s2, s1]

        s4 = PickleStep(Path.cwd(), "s4", [], {}, lambda x: None, [s2, s3])
        assert expand_dependencies([s4]) == [s4, s2, s1, s3, s2, s1]

    @staticmethod
    def test_branched_deps():
        root = PickleStep(Path.cwd(), "root", [], {}, lambda x: None, [])

        a1 = PickleStep(Path.cwd(), "a1", [], {}, lambda x: None, [root])
        a2 = PickleStep(Path.cwd(), "a2", [], {}, lambda x: None, [a1])
        a3 = PickleStep(Path.cwd(), "a3", [], {}, lambda x: None, [a2])

        b1 = PickleStep(Path.cwd(), "b1", [], {}, lambda x: None, [root])

        final = PickleStep(
            Path.cwd(), "final", [], {}, lambda x: None, [a3, b1]
        )

        assert expand_dependencies([final]) == [
            final,
            a3,
            a2,
            a1,
            root,
            b1,
            root,
        ]

        reduce_dependencies(expand_dependencies([final]))


class TestDependencyReduction:
    @staticmethod
    def test_empty_deps():
        assert reduce_dependencies([]) == []

    @staticmethod
    def test_linear_deps():
        s1 = PickleStep(Path.cwd(), "s1", [], {}, lambda x: None, [])
        s2 = PickleStep(Path.cwd(), "s2", [], {}, lambda x: None, [s1])
        s3 = PickleStep(Path.cwd(), "s3", [], {}, lambda x: None, [s2])
        s4 = PickleStep(Path.cwd(), "s4", [], {}, lambda x: None, [s2, s3])
        assert reduce_dependencies([s4, s2, s1, s3, s2, s1]) == [
            s4,
            s3,
            s2,
            s1,
        ]

    @staticmethod
    def test_branched_deps():
        root = PickleStep(Path.cwd(), "root", [], {}, lambda x: None, [])

        a1 = PickleStep(Path.cwd(), "a1", [], {}, lambda x: None, [root])
        a2 = PickleStep(Path.cwd(), "a2", [], {}, lambda x: None, [a1])
        a3 = PickleStep(Path.cwd(), "a3", [], {}, lambda x: None, [a2])

        b1 = PickleStep(Path.cwd(), "b1", [], {}, lambda x: None, [root])

        final = PickleStep(
            Path.cwd(), "final", [], {}, lambda x: None, [a3, b1]
        )

        assert reduce_dependencies(
            [
                final,
                a3,
                a2,
                a1,
                b1,
                root,
            ]
        ) == [
            final,
            a3,
            a2,
            a1,
            b1,
            root,
        ]
