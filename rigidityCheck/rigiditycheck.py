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
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def Check(index: int, element_object_list: List[ElementObject], visited=[]) -> ElementStatus:
        # -------------------- grounded: only fixed is cannot be determined --------------------#
        grounded_status = GroundedChecker.Check(index, element_object_list)
        if (
            grounded_status == ElementStatus.unassembled
            or grounded_status == ElementStatus.float
            or grounded_status == ElementStatus.rotate
        ):
            return grounded_status

        # -------------------- directly grounded --------------------#
        if element_object_list[index].is_grounded:
            return ElementStatus.fixed

        # -------------------- judge the fixed constrain num --------------------#
        fix_constrain_num = 0
        element_object = element_object_list[index]
        for neighbor_index in element_object.assembled_elements:
            if neighbor_index in visited:
                continue
            neighbor_status = TwoFixConstrainChecker.Check(neighbor_index, element_object_list, visited + [index])
            if neighbor_status == ElementStatus.fixed:
                fix_constrain_num += 1

        if fix_constrain_num >= 2:
            return ElementStatus.fixed
        else:
            return ElementStatus.rotate


class AlgebraicChecker(object):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def BuildRigidityMatrix(
        assembled: List[int],
        element_object_list: List[ElementObject],
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


        # Preserve every physical coupler segment.
        for coupler in couplers:
            coupler_vertex_1, coupler_vertex_2 = (
                couplers_dict[coupler]
            )

            const_length_constrains_vertex.append(
                [
                    coupler_vertex_1,
                    coupler_vertex_2,
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

        # # -------------------- collinear constraints --------------------#
        # collinear_constrains_vertex = []

        # for coupler in couplers:
        #     coupler_vertex_1, coupler_vertex_2 = couplers_dict[coupler]

        #     if coupler_vertex_1.element_index in elements_dict:
        #         vertex_start = elements_dict[
        #             coupler_vertex_1.element_index
        #         ][0]
        #         vertex_end = elements_dict[
        #             coupler_vertex_1.element_index
        #         ][1]

        #         collinear_constrains_vertex.append(
        #             [
        #                 vertex_start,
        #                 coupler_vertex_1,
        #                 vertex_end,
        #             ]
        #         )

        #     if coupler_vertex_2.element_index in elements_dict:
        #         vertex_start = elements_dict[
        #             coupler_vertex_2.element_index
        #         ][0]
        #         vertex_end = elements_dict[
        #             coupler_vertex_2.element_index
        #         ][1]

        #         collinear_constrains_vertex.append(
        #             [
        #                 vertex_start,
        #                 coupler_vertex_2,
        #                 vertex_end,
        #             ]
        #         )

        # K_collinear = AlgebraicChecker.CreateCollinearConstrains(
        #     collinear_constrains_vertex,
        #     vertex_num,
        # )
        
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
                grounded_constrains_vertex.append(
                    elements_dict[index][0]
                )
                grounded_constrains_vertex.append(
                    elements_dict[index][1]
                )

        K_grounded = AlgebraicChecker.CreateGroundedConstrains(
            grounded_constrains_vertex,
            vertex_num,
        )

        # -------------------- combine matrices --------------------#
        K = K_const_length

        if K_rotation is not None:
            K = np.vstack((K, K_rotation))

        if K_collinear is not None:
            K = np.vstack((K, K_collinear))

        if K_grounded is not None:
            K = np.vstack((K, K_grounded))

        return RigidityMatrixResult(
            matrix=K,
            vertex_list=vertex_list,
            elements_dict=elements_dict,
        )

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
    def Check(index: int, assembled: List[int], element_object_list: List[ElementObject]) -> ElementStatus:
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
        two_fix_status = TwoFixConstrainChecker.Check(index, element_object_list, visited=[])
        if (
            two_fix_status == ElementStatus.unassembled
            or two_fix_status == ElementStatus.float
            or two_fix_status == ElementStatus.fixed
        ):
            return two_fix_status

        matrix_result = AlgebraicChecker.BuildRigidityMatrix(
            assembled,
            element_object_list,
        )

        K = matrix_result.matrix
        dof = K.shape[1]

        if np.linalg.matrix_rank(K) == dof:
            return ElementStatus.fixed

        return ElementStatus.rotate

    @staticmethod
    def CreateVertex(vertex_list: List[Vertex], point: List[float], element_index: int = -1) -> Vertex:
        new_id = len(vertex_list)
        new_vertex = Vertex(id=new_id, point=point, element_index=element_index)
        vertex_list.append(new_vertex)
        return new_vertex

    @staticmethod
    def CreateConstLengthConstrains(constrains_vertex: List[List[Vertex]], vertex_num: int) -> Union[np.ndarray, None]:
        K = None
        for vertices in constrains_vertex:
            vertex_i: Vertex = vertices[0]
            vertex_j: Vertex = vertices[1]

            p_i = np.array(vertex_i.point).reshape((3, 1))
            p_j = np.array(vertex_j.point).reshape((3, 1))

            i = vertex_i.id
            j = vertex_j.id

            K_row = np.zeros((1, vertex_num * 3))
            K_row[0, 3 * i : 3 * i + 3] = p_i.transpose() - p_j.transpose()
            K_row[0, 3 * j : 3 * j + 3] = -(p_i.transpose() - p_j.transpose())
            if K is None:
                K = K_row
            else:
                K = np.vstack((K, K_row))

        return K

    @staticmethod
    def CreateRotationConstrains(constrains_vertex: List[List[Vertex]], vertex_num: int) -> Union[np.ndarray, None]:
        K = None
        for vertices in constrains_vertex:
            vertex_i: Vertex = vertices[0]
            vertex_j: Vertex = vertices[1]
            vertex_k: Vertex = vertices[2]

            p_i = np.array(vertex_i.point).reshape((3, 1))
            p_j = np.array(vertex_j.point).reshape((3, 1))
            p_k = np.array(vertex_k.point).reshape((3, 1))

            i = vertex_i.id
            j = vertex_j.id
            k = vertex_k.id

            K_row = np.zeros((1, vertex_num * 3))
            K_row[0, 3 * i : 3 * i + 3] = (p_j - p_k).transpose()
            K_row[0, 3 * j : 3 * j + 3] = ((p_i - p_j) - (p_j - p_k)).transpose()
            K_row[0, 3 * k : 3 * k + 3] = -(p_i - p_j).transpose()

            if K is None:
                K = K_row
            else:
                K = np.vstack((K, K_row))

        return K

    @staticmethod
    def CreateCollinearConstrains(constrains_vertex: List[List[Vertex]], vertex_num: int) -> Union[np.ndarray, None]:
        K = None
        for vertices in constrains_vertex:
            vertex_i: Vertex = vertices[0]
            vertex_j: Vertex = vertices[1]
            vertex_k: Vertex = vertices[2]

            p_i = np.array(vertex_i.point).reshape((3, 1))
            p_j = np.array(vertex_j.point).reshape((3, 1))
            p_k = np.array(vertex_k.point).reshape((3, 1))

            i = vertex_i.id
            j = vertex_j.id
            k = vertex_k.id

            K_block = np.zeros((3, vertex_num * 3))
            K_block[:, 3 * i : 3 * i + 3] = -AlgebraicChecker.CreateAntisymmetricMat(p_j - p_k)
            K_block[:, 3 * j : 3 * j + 3] = AlgebraicChecker.CreateAntisymmetricMat(p_i - p_k)
            K_block[:, 3 * k : 3 * k + 3] = -AlgebraicChecker.CreateAntisymmetricMat(p_i - p_j)

            if K is None:
                K = K_block
            else:
                K = np.vstack((K, K_block))

        return K

    @staticmethod
    def CreateGroundedConstrains(constrains_vertex: List[Vertex], vertex_num: int) -> Union[np.ndarray, None]:
        K = None
        for vertex in constrains_vertex:
            vertex: Vertex
            p_i = np.array(vertex.point).reshape((3, 1))
            i = vertex.id

            K_block = np.zeros((3, vertex_num * 3))
            K_block[:, 3 * i : 3 * i + 3] = np.eye(3)

            if K is None:
                K = K_block
            else:
                K = np.vstack((K, K_block))

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
