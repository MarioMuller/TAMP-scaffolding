# based on https://github.com/yijiangh/husky_assembly_tamp/blob/yh/dual_arm_integrate/husky_assembly_tamp/symbolic_planner/status_checker.py

from __future__ import annotations

from collections import deque, namedtuple
from typing import List, Tuple, Union

import numpy as np
try:
    from .Datastructures import ElementObject
    from .Datastructures import ElementStatus
    from .utils import closest_points_between_segments
except ImportError:
    from Datastructures import ElementObject
    from Datastructures import ElementStatus
    from utils import closest_points_between_segments

Vertex = namedtuple("Vertex", ["id", "point", "element_index"])
RigidityMatrixResult = namedtuple(
    "RigidityMatrixResult",
    [
        "matrix",
        "vertex_list",
        "elements_dict",
        "orientation_vertices",
    ],
)


class DefaultChecker(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def Check(index: int, element_object_list: List[ElementObject]) -> ElementStatus:
        return ElementStatus.fixed


class BasicChecker(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def Check(index: int, element_object_list: List[ElementObject]) -> ElementStatus:
        # -------------------- first judge assemble state --------------------#
        if element_object_list[index].status == ElementStatus.unassembled:
            return ElementStatus.unassembled

        # -------------------- second judge ground state --------------------#
        if element_object_list[index].is_grounded:
            return ElementStatus.fixed

        if len(element_object_list[index].assembled_elements) == 0:
            return ElementStatus.float
        elif len(element_object_list[index].assembled_elements) == 1:
            return ElementStatus.rotate
        else:
            return ElementStatus.fixed


class GroundedChecker(object):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def Check(index: int, element_object_list: List[ElementObject]) -> ElementStatus:

        basic_status = BasicChecker.Check(index, element_object_list)

        if basic_status == ElementStatus.fixed or basic_status == ElementStatus.unassembled:
            return basic_status

        queue = deque([index])
        visited = set([index])
        predecessor = {index: None}

        # path = []
        is_grounded = False
        while queue:
            node_index = queue.popleft()
            if element_object_list[node_index].is_grounded:
                # while node_index is not None:
                #     path.append(node_index)
                #     node_index = predecessor[node_index]
                is_grounded = True
                break
            for neighbor_index in element_object_list[node_index].assembled_elements:
                if neighbor_index not in visited:
                    queue.append(neighbor_index)
                    visited.add(neighbor_index)
                    predecessor[neighbor_index] = node_index
        # if len(path) == 0:
        #     is_grounded = False
        # else:
        #     is_grounded = True

        if is_grounded:
            return basic_status
        else:
            return ElementStatus.float

    @staticmethod
    def CheckGroundNum(index: int, element_object_list: List[ElementObject]) -> int:
        queue = deque([index])
        visited = set([index])
        ground_num = 0
        while queue:
            node_index = queue.popleft()
            if element_object_list[node_index].is_grounded:
                ground_num += 1
            for neighbor_index in element_object_list[node_index].assembled_elements:
                if neighbor_index not in visited:
                    queue.append(neighbor_index)
                    visited.add(neighbor_index)
        return ground_num

    @staticmethod
    def GetGroundPath(index: int, element_object_list: List[ElementObject]) -> List[ElementObject]:
        queue = deque([index])
        visited = set([index])
        predecessor = {index: None}

        path = []
        while queue:
            node_index = queue.popleft()
            if element_object_list[node_index].is_grounded:
                while node_index is not None:
                    path.append(node_index)
                    node_index = predecessor[node_index]
                return path[::-1]
            for neighbor_index in element_object_list[node_index].assembled_elements:
                if neighbor_index not in visited:
                    queue.append(neighbor_index)
                    visited.add(neighbor_index)
                    predecessor[neighbor_index] = node_index
        return []

    @staticmethod
    def GetTrueGroundPath(index: int, element_object_list: List[ElementObject]) -> List[ElementObject]:
        queue = deque([index])
        visited = set([index])
        predecessor = {index: None}

        path = []
        while queue:
            node_index = queue.popleft()
            if element_object_list[node_index].is_grounded:
                while node_index is not None:
                    path.append(node_index)
                    node_index = predecessor[node_index]
                return path[::-1]
            for neighbor_index in element_object_list[node_index].coupled_elements:
                if neighbor_index not in visited:
                    queue.append(neighbor_index)
                    visited.add(neighbor_index)
                    predecessor[neighbor_index] = node_index
        return []


class TwoFixConstrainChecker(object):
    @staticmethod
    def Check(
        index: int,
        element_object_list: List[ElementObject],
        visited: set[int] | None = None,
        status_cache: dict[int, ElementStatus] | None = None,
    ) -> ElementStatus:
        if visited is None:
            visited = set()

        if status_cache is None:
            status_cache = {}

        if index in status_cache:
            return status_cache[index]

        grounded_status = GroundedChecker.Check(
            index,
            element_object_list,
        )

        if grounded_status in (
            ElementStatus.unassembled,
            ElementStatus.float,
            ElementStatus.rotate,
        ):
            status_cache[index] = grounded_status
            return grounded_status

        if element_object_list[index].is_grounded:
            status_cache[index] = ElementStatus.fixed
            return ElementStatus.fixed

        if index in visited:
            return ElementStatus.rotate

        visited.add(index)

        try:
            fixed_neighbor_count = 0

            for neighbor_index in (
                element_object_list[index].assembled_elements
            ):
                neighbor_status = TwoFixConstrainChecker.Check(
                    neighbor_index,
                    element_object_list,
                    visited,
                    status_cache,
                )

                if neighbor_status == ElementStatus.fixed:
                    fixed_neighbor_count += 1

                    if fixed_neighbor_count >= 2:
                        status_cache[index] = ElementStatus.fixed
                        return ElementStatus.fixed

            status_cache[index] = ElementStatus.rotate
            return ElementStatus.rotate

        finally:
            visited.remove(index)


class AlgebraicChecker(object):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def BuildRigidityMatrix(
        assembled: List[int],
        element_object_list: List[ElementObject],
        orientation_offset_ratio: float = 0.10,
    ) -> RigidityMatrixResult:
        """
        Build the rigidity matrix for the assembled structure.

        Returns:
            matrix:
                Complete rigidity matrix K.

            vertex_list:
                Internal vertices used by the algebraic checker.

            elements_dict:
                Maps each element index to its two endpoint vertices.
        """

        # -------------------- generate vertices of elements --------------------#
        vertex_list = []
        elements_dict = {}

        for index in assembled:
            element = element_object_list[index]

            vertex_1 = AlgebraicChecker.CreateVertex(
                vertex_list,
                element.vertices[0].tolist(),
                element_index=index,
            )
            vertex_2 = AlgebraicChecker.CreateVertex(
                vertex_list,
                element.vertices[1].tolist(),
                element_index=index,
            )

            elements_dict[index] = [vertex_1, vertex_2]

        # -------------------- generate rod orientation vertices --------------------#
        #
        # A rod represented only by its two centerline endpoints has no
        # observable roll about its own axis.  Each virtual orientation vertex
        # Q lies off the centerline and introduces this missing roll degree of
        # freedom.  Q is deliberately kept separate from rod_vertices_in_order:
        # it is not a point on the split rod centerline.
        orientation_vertices = {}

        for index in assembled:
            rod_start, rod_end = elements_dict[index]

            orientation_point_1, orientation_point_2 = (
                AlgebraicChecker.CreateOrientationPoints(
                    rod_start.point,
                    rod_end.point,
                    offset_ratio=orientation_offset_ratio,
                )
            )

            orientation_vertex_1 = AlgebraicChecker.CreateVertex(
                vertex_list,
                orientation_point_1.tolist(),
                # -1 prevents external centerline visualizers from treating Q
                # as another point on this rod.  The dictionary above stores
                # the actual association with the rod.
                element_index=-1,
            )

            orientation_vertex_2 = AlgebraicChecker.CreateVertex(
                vertex_list,
                orientation_point_2.tolist(),
                # -1 prevents external centerline visualizers from treating Q
                # as another point on this rod.  The dictionary above stores
                # the actual association with the rod.
                element_index=-1,
            )

            orientation_vertices[index] = (
                orientation_vertex_1,
                orientation_vertex_2,
            )

        # -------------------- generate couplers --------------------#
        couplers = ElementObject.GetCouplers(
            assembled,
            element_object_list,
        )

        couplers_dict = {}

        for coupler in couplers:
            point_1, point_2 = closest_points_between_segments(
                element_object_list[coupler[0]].vertices,
                element_object_list[coupler[1]].vertices,
            )

            vertex_1 = AlgebraicChecker.CreateVertex(
                vertex_list,
                point_1,
                element_index=coupler[0],
            )
            vertex_2 = AlgebraicChecker.CreateVertex(
                vertex_list,
                point_2,
                element_index=coupler[1],
            )

            couplers_dict[coupler] = [vertex_1, vertex_2]
            
            
        #catch rod sections with no length
        minimum_endpoint_distance = 1e-4  # geometry is in mm

        for coupler, coupler_vertices in couplers_dict.items():
            rod_1, rod_2 = coupler
            coupler_vertex_1, coupler_vertex_2 = coupler_vertices

            for rod_index, coupler_vertex in (
                (rod_1, coupler_vertex_1),
                (rod_2, coupler_vertex_2),
            ):
                rod_start, rod_end = elements_dict[rod_index]

                coupler_point = np.asarray(
                    coupler_vertex.point,
                    dtype=float,
                )
                start_point = np.asarray(
                    rod_start.point,
                    dtype=float,
                )
                end_point = np.asarray(
                    rod_end.point,
                    dtype=float,
                )

                distance_to_start = float(
                    np.linalg.norm(coupler_point - start_point)
                )
                distance_to_end = float(
                    np.linalg.norm(coupler_point - end_point)
                )

                if min(
                    distance_to_start,
                    distance_to_end,
                ) <= minimum_endpoint_distance:
                    endpoint_name = (
                        "start"
                        if distance_to_start <= distance_to_end
                        else "end"
                    )

                    raise ValueError(
                        "Invalid scaffold geometry: "
                        f"coupler {coupler} lies at the {endpoint_name} "
                        f"of rod {rod_index}. "
                        "This would create a zero-length rod section."
                    )

        vertex_num = len(vertex_list)

        # -------------------- constant-length constraints --------------------#
        
        # -------------------- points belonging to each rod --------------------#
        coupler_vertices_by_element = {
            index: []
            for index in assembled
        }

        for coupler, coupler_vertices in couplers_dict.items():
            rod_1, rod_2 = coupler
            coupler_vertex_1, coupler_vertex_2 = coupler_vertices

            # coupler_vertex_1 lies on rod_1
            coupler_vertices_by_element[rod_1].append(
                coupler_vertex_1
            )

            # coupler_vertex_2 lies on rod_2
            coupler_vertices_by_element[rod_2].append(
                coupler_vertex_2
            )


        # -------------------- split rods at coupler points --------------------#
        rod_vertices_in_order = {}

        for index in assembled:
            rod_start, rod_end = elements_dict[index]

            start_point = np.asarray(
                rod_start.point,
                dtype=float,
            )
            end_point = np.asarray(
                rod_end.point,
                dtype=float,
            )

            rod_direction = end_point - start_point
            rod_length_squared = float(
                np.dot(rod_direction, rod_direction)
            )

            if rod_length_squared == 0.0:
                raise ValueError(
                    f"Rod {index} has zero length."
                )

            vertices_on_rod = [
                rod_start,
                *coupler_vertices_by_element[index],
                rod_end,
            ]

            def rod_parameter(vertex: Vertex) -> float:
                point = np.asarray(
                    vertex.point,
                    dtype=float,
                )

                return float(
                    np.dot(
                        point - start_point,
                        rod_direction,
                    )
                    / rod_length_squared
                )

            vertices_on_rod.sort(
                key=rod_parameter
            )

            minimum_segment_length = 1e-4  # mm

            for vertex_1, vertex_2 in zip(
                vertices_on_rod,
                vertices_on_rod[1:],
            ):
                point_1 = np.asarray(
                    vertex_1.point,
                    dtype=float,
                )
                point_2 = np.asarray(
                    vertex_2.point,
                    dtype=float,
                )

                segment_length = float(
                    np.linalg.norm(point_2 - point_1)
                )

                if segment_length <= minimum_segment_length:
                    raise ValueError(
                        "Invalid scaffold geometry: "
                        f"rod {index} contains coincident split points. "
                        f"Vertices {vertex_1.id} and {vertex_2.id} are "
                        f"{segment_length:.6g} mm apart. "
                        "A coupler may lie at a rod endpoint, or multiple "
                        "couplers may occupy the same position."
                    )

            rod_vertices_in_order[index] = vertices_on_rod


        # -------------------- constant-length constraints --------------------#
        const_length_constrains_vertex = []

        # Each rod is divided into adjacent subsegments:
        #
        # start -- coupler 1 -- coupler 2 -- ... -- end
        #
        # Preserve the length of every adjacent subsegment.
        for index in assembled:
            vertices_on_rod = rod_vertices_in_order[index]

            for vertex_1, vertex_2 in zip(
                vertices_on_rod,
                vertices_on_rod[1:],
            ):
                const_length_constrains_vertex.append(
                    [vertex_1, vertex_2]
                )


        # Give every rod an off-axis orientation frame.
        #
        # The two distances Q-A and Q-B keep Q at a fixed radius from the
        # rod axis.  Q can still rotate around A-B; that remaining motion is
        # precisely the rod's roll coordinate.
        for index in assembled:
            rod_start, rod_end = elements_dict[index]
            orientation_vertex_1, orientation_vertex_2 = (
                orientation_vertices[index]
            )

            const_length_constrains_vertex.extend(
                [
                    # Attach Q1 to the rod.
                    [orientation_vertex_1, rod_start],
                    [orientation_vertex_1, rod_end],

                    # Attach Q2 to the rod.
                    [orientation_vertex_2, rod_start],
                    [orientation_vertex_2, rod_end],

                    # Prevent Q1 and Q2 from rolling independently.
                    [orientation_vertex_1, orientation_vertex_2],
                ]
            )

        # Preserve every physical coupler segment and tie its azimuth to the
        # orientation frame of both rods.
        #
        # For coupler (rod_1, rod_2):
        #   C1 lies on rod_1, C2 lies on rod_2.
        # The distances Q1-C2 and Q2-C1 prevent the coupler from rotating
        # independently around either rod axis, while still allowing the whole
        # rod-coupler assembly to undergo a common rigid-body rotation.
        for coupler in couplers:
            rod_1, rod_2 = coupler
            coupler_vertex_1, coupler_vertex_2 = (
                couplers_dict[coupler]
            )

            rod_1_q1, rod_1_q2 = orientation_vertices[rod_1]
            rod_2_q1, rod_2_q2 = orientation_vertices[rod_2]

            const_length_constrains_vertex.extend(
                [
                    # Physical coupler length.
                    [coupler_vertex_1, coupler_vertex_2],

                    # Coupler orientation relative to rod 1.
                    [rod_1_q1, coupler_vertex_2],
                    [rod_1_q2, coupler_vertex_2],

                    # Coupler orientation relative to rod 2.
                    [rod_2_q1, coupler_vertex_1],
                    [rod_2_q2, coupler_vertex_1],
                ]
            )

        K_const_length = (
            AlgebraicChecker.CreateConstLengthConstrains(
                const_length_constrains_vertex,
                vertex_num,
            )
        )
        
        # const_length_constrains_vertex = []

        # for index in assembled:
        #     vertex_1 = elements_dict[index][0]
        #     vertex_2 = elements_dict[index][1]

        #     const_length_constrains_vertex.append(
        #         [vertex_1, vertex_2]
        #     )

        #     for coupler, coupler_vertices in couplers_dict.items():
        #         if index in coupler:
        #             vertex_2 = coupler_vertices[
        #                 1 - coupler.index(index)
        #             ]
        #             const_length_constrains_vertex.append(
        #                 [vertex_1, vertex_2]
        #             )

        # for coupler in couplers:
        #     vertex_1, vertex_2 = couplers_dict[coupler]
        #     const_length_constrains_vertex.append(
        #         [vertex_1, vertex_2]
        #     )

        # K_const_length = AlgebraicChecker.CreateConstLengthConstrains(
        #     const_length_constrains_vertex,
        #     vertex_num,
        # )

        # -------------------- rotation constraints --------------------#
        rotation_constrains_vertex = []

        for coupler in couplers:
            coupler_vertex_1, coupler_vertex_2 = couplers_dict[coupler]

            if coupler_vertex_1.element_index in elements_dict:
                segment_vertex = elements_dict[
                    coupler_vertex_1.element_index
                ][0]

                rotation_constrains_vertex.append(
                    [
                        segment_vertex,
                        coupler_vertex_1,
                        coupler_vertex_2,
                    ]
                )

            if coupler_vertex_2.element_index in elements_dict:
                segment_vertex = elements_dict[
                    coupler_vertex_2.element_index
                ][0]

                rotation_constrains_vertex.append(
                    [
                        segment_vertex,
                        coupler_vertex_2,
                        coupler_vertex_1,
                    ]
                )

        K_rotation = AlgebraicChecker.CreateRotationConstrains(
            rotation_constrains_vertex,
            vertex_num,
        )
        
        # -------------------- collinear constraints --------------------#
        collinear_constrains_vertex = []

        for index in assembled:
            vertices_on_rod = rod_vertices_in_order[index]

            # For:
            #
            # p0 -- p1 -- p2 -- p3
            #
            # create triples:
            #
            # [p0, p1, p2]
            # [p1, p2, p3]
            for vertex_1, vertex_2, vertex_3 in zip(
                vertices_on_rod,
                vertices_on_rod[1:],
                vertices_on_rod[2:],
            ):
                collinear_constrains_vertex.append(
                    [
                        vertex_1,
                        vertex_2,
                        vertex_3,
                    ]
                )

        K_collinear = (
            AlgebraicChecker.CreateCollinearConstrains(
                collinear_constrains_vertex,
                vertex_num,
            )
        )

        # -------------------- grounded constraints --------------------#
        grounded_constrains_vertex = []

        for index in assembled:
            element = element_object_list[index]

            if element.is_grounded:
                # Ground the complete rod frame.  Fixing only the two
                # centerline endpoints would still leave axial roll free.
                orientation_vertex_1, orientation_vertex_2 = (
                orientation_vertices[index]
                )

                grounded_constrains_vertex.extend(
                    [
                        elements_dict[index][0],
                        elements_dict[index][1],
                        orientation_vertex_1,
                        orientation_vertex_2,
                    ]
                )

        K_grounded = AlgebraicChecker.CreateGroundedConstrains(
            grounded_constrains_vertex,
            vertex_num,
        )

        # -------------------- combine matrices --------------------#
        matrix_blocks = [
            block
            for block in (
                K_const_length,
                K_rotation,
                K_collinear,
                K_grounded,
            )
            if block is not None
        ]

        K = np.vstack(matrix_blocks)
        
        print(f"Rigidity matrix K shape: {K.shape}")

        return RigidityMatrixResult(
            matrix=K,
            vertex_list=vertex_list,
            elements_dict=elements_dict,
            orientation_vertices=orientation_vertices,
        )

    @staticmethod
    def AnalyzeQR(
        K: np.ndarray,
        tolerance: float | None = None,
    ) -> tuple[int, tuple[np.ndarray, ...]]:
        from scipy.linalg import qr

        # K.T has shape (DOF, constraints).
        # The final columns of Q span null(K).
        Q, R, _ = qr(
            K.T,
            mode="full",
            pivoting=True,
            overwrite_a=False,
            check_finite=False,
        )

        diagonal = np.abs(np.diag(R))

        if tolerance is None:
            largest = diagonal.max() if diagonal.size else 0.0
            tolerance = (
                max(K.shape)
                * np.finfo(K.dtype).eps
                * largest
            )

        rank = int(np.count_nonzero(diagonal > tolerance))

        failure_modes = tuple(
            Q[:, column].copy()
            for column in range(rank, Q.shape[1])
        )

        return rank, failure_modes

    @staticmethod
    def AnalyzeSVD(
        K: np.ndarray,
        tolerance: float | None = None,
    ) -> tuple[int, tuple[np.ndarray, ...]]:
        _, singular_values, vh = np.linalg.svd(
            K,
            full_matrices=False,
        )


        import os

        print(
            "OPENBLAS_NUM_THREADS:",
            os.environ.get("OPENBLAS_NUM_THREADS"),
        )


        if tolerance is None:
            largest = (
                singular_values[0]
                if singular_values.size
                else 0.0
            )
            tolerance = (
                max(K.shape)
                * np.finfo(K.dtype).eps
                * largest
            )

        rank = int(np.count_nonzero(singular_values > tolerance))

        failure_modes = tuple(
            mode.copy()
            for mode in vh[rank:]
        )

        return rank, failure_modes
        
    @staticmethod
    def GetNullspaceModes(
        K: np.ndarray,
        method: str = "svd",
        tolerance: float | None = None,
    ) -> list[np.ndarray]:
        """
        Return an orthonormal basis of the null space of K.

        Args:
            K:
                Rigidity matrix.

            method:
                "svd" or "qr".

            tolerance:
                Numerical threshold for rank detection.
        """

        method = method.lower()

        if method == "svd":
            _, singular_values, vh = np.linalg.svd(
                K,
                full_matrices=True,
            )

            if tolerance is None:
                largest = (
                    singular_values[0]
                    if len(singular_values)
                    else 0.0
                )
                tolerance = (
                    max(K.shape)
                    * np.finfo(float).eps
                    * largest
                )

            rank = int(
                np.sum(singular_values > tolerance)
            )

            return [
                mode.copy()
                for mode in vh[rank:]
            ]

        if method == "qr":
            from scipy.linalg import qr

            # K.T = Q R with column pivoting.
            Q, R, _ = qr(
                K.T,
                mode="full",
                pivoting=True,
            )

            diagonal = np.abs(np.diag(R))

            if tolerance is None:
                largest = (
                    diagonal.max()
                    if diagonal.size
                    else 0.0
                )
                tolerance = (
                    max(K.shape)
                    * np.finfo(float).eps
                    * largest
                )

            rank = int(
                np.sum(diagonal > tolerance)
            )

            # Columns Q[:, rank:] span null(K).
            return [
                Q[:, column].copy()
                for column in range(rank, Q.shape[1])
            ]

        raise ValueError(
            f"Unknown null-space method: {method!r}. "
            "Use 'svd' or 'qr'."
        )

    
    @staticmethod
    def Check(index: int, assembled: List[int], element_object_list: List[ElementObject], matrix_is_full_rank: bool | None = None, status_cache: dict[int, ElementStatus] | None = None,) -> ElementStatus:
        """
        Check stability of element given by index.

        Params:
            index (int): index of current element
            assembled ([int]): indices of assembled elements excluding current element
            element_object_list ([ElementObject]): list of ElementObject

        Returns:
            ElementStatus: status of current element
        """
        if index not in assembled:
            assembled.append(index)

        # -------------------- grounded: only rotate is cannot be determined --------------------#
        two_fix_status = TwoFixConstrainChecker.Check(
            index,
            element_object_list,
            status_cache=status_cache,
        )
        if (
            two_fix_status == ElementStatus.unassembled
            or two_fix_status == ElementStatus.float
            or two_fix_status == ElementStatus.fixed
        ):
            return two_fix_status
        
         # Backward-compatible fallback for isolated uses of this method.
        if matrix_is_full_rank is None:
            matrix_result = AlgebraicChecker.BuildRigidityMatrix(
                assembled,
                element_object_list,
            )
            K = matrix_result.matrix
            matrix_is_full_rank = (
                np.linalg.matrix_rank(K) == K.shape[1]
            )


        return (
                ElementStatus.fixed
                if matrix_is_full_rank
                else ElementStatus.rotate
            )

    @staticmethod
    def CreateOrientationPoints(
        rod_start: List[float],
        rod_end: List[float],
        offset_ratio: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray]:
        start = np.asarray(rod_start, dtype=float)
        end = np.asarray(rod_end, dtype=float)
        """Creates two deterministic virtual point away from a rod axis.

        The points are placed at the rod midpoint plus a perpendicular offset.
        The remaining motion around the rod axis represents the rod's roll.
        """
        if offset_ratio <= 0.0:
            raise ValueError(
                "orientation_offset_ratio must be greater than zero."
            )

        start = np.asarray(rod_start, dtype=float)
        end = np.asarray(rod_end, dtype=float)

        rod_vector = end - start
        rod_length = float(np.linalg.norm(rod_vector))

        if rod_length == 0.0:
            raise ValueError("Cannot orient a zero-length rod.")

        rod_axis = rod_vector / rod_length

        # Select the world basis direction least parallel to the rod.  This
        # maximizes the cross-product magnitude and avoids a fragile special
        # case for vertical rods.
        world_axes = np.eye(3)
        reference_axis = world_axes[
            int(np.argmin(np.abs(world_axes @ rod_axis)))
        ]

        radial_1 = np.cross(rod_axis, reference_axis)
        radial_1 /= np.linalg.norm(radial_1)

        # Second radial direction, perpendicular to both the rod and radial_1.
        radial_2 = np.cross(rod_axis, radial_1)
        radial_2 /= np.linalg.norm(radial_2)

        midpoint = 0.5 * (start + end)
        offset = offset_ratio * rod_length

        q1 = midpoint + offset * radial_1
        q2 = midpoint + offset * radial_2

        return q1, q2

    @staticmethod
    def CreateVertex(vertex_list: List[Vertex], point: List[float], element_index: int = -1) -> Vertex:
        new_id = len(vertex_list)
        new_vertex = Vertex(id=new_id, point=point, element_index=element_index)
        vertex_list.append(new_vertex)
        return new_vertex

    @staticmethod
    def CreateConstLengthConstrains(constrains_vertex: List[List[Vertex]], vertex_num: int) -> np.ndarray | None: 
        
        constraint_count = len(constrains_vertex)
        
        if constraint_count == 0:
            return None
        
        K = np.zeros((constraint_count, vertex_num * 3), dtype=float)
        
        for vertices in constrains_vertex:
            vertex_i: Vertex = vertices[0]
            vertex_j: Vertex = vertices[1]

            p_i = np.array(vertex_i.point).reshape((3, 1))
            p_j = np.array(vertex_j.point).reshape((3, 1))

            i = vertex_i.id
            j = vertex_j.id

        for row, (vertex_i, vertex_j) in enumerate(constrains_vertex):
            difference = (
                np.asarray(vertex_i.point, dtype=float)
                - np.asarray(vertex_j.point, dtype=float)
            )

            i_start = 3 * vertex_i.id
            j_start = 3 * vertex_j.id

            K[row, i_start:i_start + 3] = difference
            K[row, j_start:j_start + 3] = -difference

        return K

    @staticmethod
    def CreateRotationConstrains(constrains_vertex: List[List[Vertex]], vertex_num: int) -> np.ndarray | None:
        
        constraint_count = len(constrains_vertex)
        
        if constraint_count == 0:
            return None
        
        K = np.zeros((constraint_count, vertex_num * 3), dtype=float)
        
        for row, vertices in enumerate(constrains_vertex):
            vertex_i: Vertex = vertices[0]
            vertex_j: Vertex = vertices[1]
            vertex_k: Vertex = vertices[2]

            p_i = np.asarray(vertex_i.point, dtype=float)
            p_j = np.asarray(vertex_j.point, dtype=float)
            p_k = np.asarray(vertex_k.point, dtype=float)

            i_start = 3 * vertex_i.id
            j_start = 3 * vertex_j.id
            k_start = 3 * vertex_k.id

            K[row, i_start:i_start + 3] = (p_j - p_k)
            K[row, j_start:j_start + 3] = ((p_i - p_j) - (p_j - p_k))
            K[row, k_start:k_start + 3] = -(p_i - p_j)

        return K

    @staticmethod
    def CreateCollinearConstrains(constrains_vertex: List[List[Vertex]], vertex_num: int) -> np.ndarray | None:
                
        constraint_count = len(constrains_vertex)
        
        if constraint_count == 0:
            return None
        
        K = np.zeros((3 * constraint_count, vertex_num * 3), dtype=float)
        
        for constraint_index, (vertex_i, vertex_j, vertex_k) in enumerate(constrains_vertex):
            p_i = np.asarray(vertex_i.point, dtype=float)
            p_j = np.asarray(vertex_j.point, dtype=float)
            p_k = np.asarray(vertex_k.point, dtype=float)
            
            row_start = 3 * constraint_index
            row_slice = slice(row_start, row_start + 3)

            i_start = 3 * vertex_i.id
            j_start = 3 * vertex_j.id
            k_start = 3 * vertex_k.id

            
            # Fill in the corresponding rows in K
            K[row_slice, i_start:i_start + 3] = (-AlgebraicChecker.CreateAntisymmetricMat(p_j - p_k))
            K[row_slice, j_start:j_start + 3] = (AlgebraicChecker.CreateAntisymmetricMat(p_i - p_k))
            K[row_slice, k_start:k_start + 3] = (-AlgebraicChecker.CreateAntisymmetricMat(p_i - p_j))
            
        return K

    @staticmethod
    def CreateGroundedConstrains(constrains_vertex: List[Vertex], vertex_num: int) -> np.ndarray | None:
        
        constraint_count = len(constrains_vertex)
        
        if constraint_count == 0:
            return None
        
        K = np.zeros((3*constraint_count, vertex_num * 3), dtype=float)
        
        identity_matrix = np.eye(3)
        
        for constraint_index, vertex in enumerate(constrains_vertex):
            row_start = 3 * constraint_index
            column_start = 3 * vertex.id

            K[row_start:row_start + 3, column_start:column_start + 3] = identity_matrix

        return K

    @staticmethod
    def CreateAntisymmetricMat(vec: np.ndarray) -> np.ndarray:
        vec = vec.reshape((3,))

        x = vec[0]
        y = vec[1]
        z = vec[2]

        mat = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return mat

    @staticmethod
    def LookupVertex(vertex_list: List[Vertex], vertex_id: int) -> Union[Vertex, None]:
        for vertex in vertex_list:
            if vertex.id == vertex_id:
                return vertex
        return None