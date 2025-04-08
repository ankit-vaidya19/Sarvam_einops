import re
from typing import List, Tuple, Dict, Any, Set
import numpy as np


class EinopsError(ValueError):
    pass


def check_ellipsis(pattern: str) -> bool:
    """Checks if the pattern segment has valid ellipsis usage.

    Args:
        pattern: A segment of the einops pattern.

    Raises:
        EinopsError: If ellipsis usage is invalid.

    Returns:
        bool: True if ellipsis exist and are valid, False if no ellipsis exist.
    """
    if pattern.count("...") == 0:
        return False
    if pattern.count("...") != 1 or pattern.count(".") != 3:
        if any(c == "." for c in pattern.replace("...", "")):
            raise EinopsError("Character '.' is reserved for ellipsis '...'")
        raise EinopsError("Pattern must contain at most one ellipsis '...' per side.")
    return True


def check_parenthesis(pattern: str) -> bool:
    """Checks if the pattern segment has valid parenthesis usage.

    Args:
        pattern: A segment of the einops pattern.

    Raises:
        EinopsError: If parenthesis usage is invalid (unbalanced, nested, empty, single item).

    Returns:
        bool: True if parentheses exist and are valid, False if no parentheses exist.
    """
    if pattern.count("(") == 0 and pattern.count(")") == 0:
        return False
    if pattern.count("(") != pattern.count(")"):
        raise EinopsError("Unmatched parentheses in pattern.")
    nesting_level = 0
    for char in pattern:
        if char == "(":
            nesting_level += 1
            if nesting_level > 1:
                raise EinopsError("Nested parentheses are not allowed.")
        elif char == ")":
            if nesting_level == 0:
                raise EinopsError(
                    "Invalid parenthesis nesting: closing bracket without opening."
                )
            nesting_level -= 1
    if nesting_level != 0:
        raise EinopsError("Invalid parenthesis nesting.")

    matches = re.findall(r"\(([^()]*?)\)", pattern)
    if not matches and (pattern.count("(") > 0):
        raise EinopsError("Parentheses found but content could not be parsed.")

    for content in matches:
        components = content.strip().split()
        if len(components) < 2:
            raise EinopsError(
                f"Parentheses group '({content})' must contain at least two identifiers."
            )
        for comp in components:
            if not re.fullmatch(r"[a-zA-Z0-9_]+", comp):
                raise EinopsError(
                    f"Invalid identifier '{comp}' found within parentheses '({content})'. Identifiers must be alphanumeric or underscore."
                )
    return True


def parse_parenthesis(composite_axis: str) -> List[str]:
    """Extracts component identifiers from a composite axis string.

    Args:
        composite_axis: The composite axis string, e.g., "(h w)".

    Returns:
        List[str]: List of component identifiers, e.g., ["h", "w"].

    Raises:
        EinopsError: If the input string is not a valid composite axis format.
    """
    match = re.fullmatch(r"\(\s*([^()]+?)\s*\)", composite_axis)
    if not match:
        raise ValueError(f"Invalid composite axis format: {composite_axis}")
    content = match.group(1).strip()
    components = content.split()
    return components


def parse_axis(axes: List[str]) -> Dict[str, Dict[str, Any]]:
    """Parses a list of axis strings into a structured dictionary.

    Args:
        axes: List of axis identifiers from one side of the pattern.

    Returns:
        Dict[str, Dict]: Dictionary mapping axis identifiers to their info
                         (class, position, components).
    """
    axes_dict = {}
    has_ellipsis = False
    for i, axis in enumerate(axes):
        axis_info: Dict[str, Any] = {"position": i, "axis_name": axis}
        if axis == "...":
            if has_ellipsis:
                raise EinopsError(
                    "Pattern cannot contain more than one ellipsis per side."
                )
            axis_info["class"] = "ellipsis"
            has_ellipsis = True
        elif axis.startswith("(") and axis.endswith(")"):
            components = parse_parenthesis(axis)
            axis_info["class"] = "composite"
            axis_info["components"] = components
        elif axis == "1":
            axis_info["class"] = "literal"
            axis_info["value"] = 1
        elif re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", axis):
            axis_info["class"] = "simple"
        elif axis.isdigit():
            raise EinopsError(
                f"Numeric literals other than '1' are not supported: '{axis}'"
            )
        else:
            raise EinopsError(f"Invalid axis identifier format: '{axis}'")
        axes_dict[f"{axis}_{i}"] = axis_info
    return axes_dict


def check_and_compare_sides(
    input_axes_list: List[str], output_axes_list: List[str], axes_length_keys: Set[str]
) -> None:
    """Validates and compares the left and right sides of the pattern.

    Args:
        input_pattern: Left side of the pattern.
        output_pattern: Right side of the pattern.

    Raises:
        EinopsError: If sides are empty, ellipsis mismatch, or identifier mismatch.
    """
    input_pattern_str = " ".join(input_axes_list)
    output_pattern_str = " ".join(output_axes_list)

    if not input_pattern_str or not output_pattern_str:
        raise EinopsError("Pattern cannot have an empty side.")

    input_has_ellipsis = "..." in input_axes_list
    output_has_ellipsis = "..." in output_axes_list
    if input_has_ellipsis != output_has_ellipsis:
        raise EinopsError("Ellipsis '...' must appear on both sides or neither.")

    check_parenthesis(input_pattern_str)
    check_parenthesis(output_pattern_str)

    input_components = set()
    input_literal_1_count = 0
    for axis in input_axes_list:
        if axis == "1":
            input_literal_1_count += 1
        elif axis == "...":
            continue
        elif axis.startswith("("):
            input_components.update(parse_parenthesis(axis))
        elif re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", axis):
            input_components.add(axis)

    output_components = set()
    for axis in output_axes_list:
        if axis == "1":
            continue
        elif axis == "...":
            continue
        elif axis.startswith("("):
            output_components.update(parse_parenthesis(axis))
        elif re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", axis):
            output_components.add(axis)

    new_on_output = output_components - input_components
    if new_on_output:
        if input_literal_1_count == 0:
            raise EinopsError(
                f"Output introduces new axes {new_on_output} but input has no literal '1' to replace."
            )
        if len(new_on_output) > input_literal_1_count:
            pass
        for axis in new_on_output:
            if axis not in axes_length_keys:
                raise EinopsError(
                    f"Size for newly introduced axis '{axis}' (replacing '1') must be specified via axes_lengths."
                )
    carried_over_output = output_components - new_on_output
    missing = carried_over_output - input_components
    if missing:
        raise EinopsError(f"Output axes {missing} are not found on the input side.")


def pattern_parser(pattern: str) -> Tuple[List[str], List[str]]:
    """Parses the full einops pattern string into left and right axis lists.

    Args:
        pattern: The complete einops pattern string (e.g., "b c h w -> b (h w) c").

    Raises:
        EinopsError: If format is invalid (no '->', multiple '->', parsing errors).

    Returns:
        Tuple[List[str], List[str]]: Tuple containing the list of axes for the
                                     left side and the right side.
    """
    if not isinstance(pattern, str) or not pattern:
        raise EinopsError("Pattern must be a non-empty string.")
    if "->" not in pattern:
        raise EinopsError("Pattern must contain '->' separator.")
    split = pattern.split("->")
    if len(split) != 2:
        raise EinopsError("Pattern must contain exactly one '->' separator.")

    left_str = split[0].strip()
    right_str = split[1].strip()

    re_axis_token = r"\.\.\.|[a-zA-Z_][a-zA-Z0-9_]*|\(\s*[a-zA-Z_][a-zA-Z0-9_]*(?:\s+[a-zA-Z_][a-zA-Z0-9_]*)+\s*\)|\d+"

    try:
        left_axes = [match.group(0) for match in re.finditer(re_axis_token, left_str)]
        right_axes = [match.group(0) for match in re.finditer(re_axis_token, right_str)]
        left_tokens_joined = "".join([re.sub(r"\s+", "", token) for token in left_axes])
        right_tokens_joined = "".join(
            [re.sub(r"\s+", "", token) for token in right_axes]
        )
        expected_left = re.sub(r"\s+", "", left_str)
        expected_right = re.sub(r"\s+", "", right_str)
        if left_tokens_joined != expected_left:
            raise EinopsError(
                f"Could not fully parse left side tokens: '{left_str}' -> {left_axes}"
            )
        if right_tokens_joined != expected_right:
            raise EinopsError(
                f"Could not fully parse right side tokens: '{right_str}' -> {right_axes}"
            )
    except Exception as e:
        raise EinopsError(f"Failed to parse pattern tokens: {e}")

    for axes_list in [left_axes, right_axes]:
        for axis in axes_list:
            if axis == "...":
                continue
            if axis.startswith("(") and axis.endswith(")"):
                try:
                    comps = parse_parenthesis(axis)
                    for comp in comps:
                        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", comp):
                            raise EinopsError(
                                f"Invalid component '{comp}' inside parenthesis '{axis}'."
                            )
                except ValueError as e:
                    raise EinopsError(f"Invalid parenthesis group '{axis}': {e}")
                continue
            if axis == "1":
                continue
            if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", axis):
                continue
            if axis.isdigit():
                raise EinopsError(
                    f"Numeric literal '{axis}' is not supported (only '1')."
                )
            raise EinopsError(f"Invalid token '{axis}' found in pattern.")

    return left_axes, right_axes


def axis_mapper(
    tensor_shape: Tuple[int, ...],
    input_axes_info: Dict[str, Dict],
    axes_lengths: Dict[str, int],
) -> Dict[str, Any]:
    """Computes dimensions for all axes based on tensor shape, pattern, and provided lengths.

    Args:
        tensor_shape: Shape of the input tensor.
        input_axes_info: Dictionary with input axis information.
        axes_lengths: Dictionary with user-specified axis lengths.

    Raises:
        EinopsError: If dimensions conflict, cannot be resolved, or pattern mismatches shape.

    Returns:
        Dict: A dictionary containing:
            - 'dims': Mapping from all unique axis names (simple & composite components) to their sizes.
            - 'ellipsis_ndim': Number of dimensions covered by the ellipsis (0 if no ellipsis).
            - 'input_expanded_axis_names': List of axis names corresponding to tensor dimensions
              after conceptually flattening composite axes (e.g., ['b', 'c', 'h', 'w'] or
              ['_ell_0', '_ell_1', 'c']). Ellipsis dimensions are named '_ell_N'.
    """
    resolved_dims = {}
    all_input_components = set()
    has_ellipsis_on_input = False

    for axis_name, length in axes_lengths.items():
        if not isinstance(length, int) or length <= 0:
            raise EinopsError(
                f"Provided dimension for axis '{axis_name}' must be a positive integer."
            )
        resolved_dims[axis_name] = length

    for axis_key, info in input_axes_info.items():
        if info["class"] == "simple":
            all_input_components.add(info["axis_name"])
        elif info["class"] == "composite":
            all_input_components.update(info["components"])
        elif info["class"] == "ellipsis":
            has_ellipsis_on_input = True

    tensor_ndim = len(tensor_shape)
    pattern_explicit_ndim = sum(
        1 for info in input_axes_info.values() if info["class"] != "ellipsis"
    )
    ellipsis_ndim = 0
    if has_ellipsis_on_input:
        if tensor_ndim < pattern_explicit_ndim:
            raise EinopsError(
                f"Input tensor has {tensor_ndim} dimensions, but pattern expects at least {pattern_explicit_ndim} non-ellipsis dimensions."
            )
        ellipsis_ndim = tensor_ndim - pattern_explicit_ndim
    elif tensor_ndim != pattern_explicit_ndim:
        raise EinopsError(
            f"Input tensor has {tensor_ndim} dimensions, but pattern expects {pattern_explicit_ndim} dimensions (no ellipsis)."
        )

    unresolved_components = set(all_input_components) - set(resolved_dims.keys())
    composite_axes_to_solve = {}
    input_expanded_axis_names = []

    sorted_input = sorted(input_axes_info.items(), key=lambda item: item[1]["position"])
    current_tensor_idx = 0
    for _, info in sorted_input:
        if info["class"] == "ellipsis":
            for i in range(ellipsis_ndim):
                input_expanded_axis_names.append(f"_ell_{i}")
            current_tensor_idx += ellipsis_ndim
            continue
        if current_tensor_idx >= tensor_ndim:
            raise EinopsError(
                f"Pattern structure error accessing tensor index {current_tensor_idx} (tensor shape {tensor_shape})."
            )
        current_dim_size = tensor_shape[current_tensor_idx]
        if info["class"] == "simple":
            token = info["axis_name"]
            input_expanded_axis_names.append(token)
            if token in resolved_dims:
                if resolved_dims[token] != current_dim_size:
                    raise EinopsError(
                        f"Dimension mismatch for axis '{token}': pattern implies size {current_dim_size} but resolved size is {resolved_dims[token]}."
                    )
            else:
                resolved_dims[token] = current_dim_size
                unresolved_components.discard(token)
        elif info["class"] == "literal":
            input_expanded_axis_names.append("1")
            if current_dim_size != 1:
                raise EinopsError(
                    f"Dimension mismatch for literal '1' at position {info['position']}: tensor shape has {current_dim_size}, expected 1."
                )
            resolved_dims["1"] = 1
        elif info["class"] == "composite":
            comps = info["components"]
            input_expanded_axis_names.extend(comps)
            composite_axes_to_solve[info["axis_name"] + f"_{info['position']}"] = {
                "total_size": current_dim_size,
                "components": list(comps),
            }
            if all(c in resolved_dims for c in comps):
                prod_val = np.prod([resolved_dims[c] for c in comps], dtype=np.int64)
                if prod_val != current_dim_size:
                    raise EinopsError(
                        f"Composite axis '{comps}' mismatch: product of components {prod_val} != tensor dimension {current_dim_size}."
                    )
                unresolved_components.difference_update(comps)
        current_tensor_idx += 1

    made_progress = True
    passes = 0
    max_passes = len(composite_axes_to_solve) + 2
    while unresolved_components and made_progress and passes < max_passes:
        made_progress = False
        passes += 1
        for key, comp_info in list(composite_axes_to_solve.items()):
            total_size = comp_info["total_size"]
            comps = comp_info["components"]
            unknown = [c for c in comps if c not in resolved_dims]
            if not unknown:
                prod_val = np.prod([resolved_dims[c] for c in comps], dtype=np.int64)
                if prod_val != total_size:
                    raise EinopsError(
                        f"Composite axis '{comps}' mismatch: product {prod_val} != tensor dimension {total_size}."
                    )
            elif len(unknown) == 1:
                unknown_comp = unknown[0]
                known_product = np.prod(
                    [resolved_dims[c] for c in comps if c != unknown_comp],
                    dtype=np.int64,
                )
                if known_product <= 0 or total_size % known_product != 0:
                    raise EinopsError(
                        f"Cannot deduce size for '{unknown_comp}' in composite {comps}: total size {total_size} not divisible by {known_product}."
                    )
                resolved_dims[unknown_comp] = total_size // known_product
                made_progress = True
                unresolved_components.discard(unknown_comp)

    if unresolved_components:
        raise EinopsError(
            f"Could not resolve dimensions for axes: {unresolved_components}. Provide explicit sizes or check the pattern."
        )

    return {
        "dims": resolved_dims,
        "ellipsis_ndim": ellipsis_ndim,
        "input_expanded_axis_names": input_expanded_axis_names,
    }


def apply_operation(
    tensor: np.ndarray,
    input_axes_info: Dict[str, Dict],
    output_axes_info: Dict[str, Dict],
    mapper_result: Dict[str, Any],
) -> np.ndarray:
    """Performs the tensor rearrangement using reshape and transpose based on parsed info.

    Args:
        tensor: Input tensor.
        input_axis_info: Parsed info for input axes.
        output_axis_info: Parsed info for output axes.
        mapper_result: The dictionary returned by axis_mapper, containing 'dims', 'ellipsis_ndim', 'input_expanded_axis_names'.

    Returns:
        np.ndarray: The rearranged tensor.
    """
    resolved_dims = mapper_result["dims"]
    ellipsis_ndim = mapper_result["ellipsis_ndim"]
    input_expanded_axis_names = mapper_result["input_expanded_axis_names"]

    input_identifiers = set()
    for info in input_axes_info.values():
        if info["class"] == "simple":
            input_identifiers.add(info["axis_name"])
        elif info["class"] == "composite":
            input_identifiers.update(info["components"])

    output_identifiers = set()
    output_comps = set()
    for info in output_axes_info.values():
        if info["class"] == "simple":
            output_identifiers.add(info["axis_name"])
            output_comps.add(info["axis_name"])
        elif info["class"] == "composite":
            output_identifiers.add(info["axis_name"])
            output_comps.update(info["components"])
    new_output_axes = output_comps - input_identifiers

    input_reshape_target = []
    original_idx = 0
    sorted_input = sorted(input_axes_info.items(), key=lambda item: item[1]["position"])
    for _, info in sorted_input:
        if info["class"] == "ellipsis":
            slice_dims = tensor.shape[original_idx : original_idx + ellipsis_ndim]
            input_reshape_target.extend(slice_dims)
            original_idx += ellipsis_ndim
        elif info["class"] == "composite":
            comps = info["components"]
            sizes = [resolved_dims[c] for c in comps]
            input_reshape_target.extend(sizes)
            original_idx += 1
        elif info["class"] == "literal":
            input_reshape_target.append(1)
            original_idx += 1
        else:
            token = info["axis_name"]
            input_reshape_target.append(resolved_dims[token])
            original_idx += 1

    if tuple(input_reshape_target) != tensor.shape:
        try:
            tensor = tensor.reshape(input_reshape_target)
        except ValueError as e:
            raise EinopsError(
                f"Reshape error: expected shape {input_reshape_target} vs input {tensor.shape}. Error: {e}"
            )

    transpose_order = []
    final_target_shape = []
    output_expanded_names = []
    literal_one_indices = [
        i for i, name in enumerate(input_expanded_axis_names) if name == "1"
    ]
    literal_one_usage = {idx: False for idx in literal_one_indices}

    sorted_output = sorted(
        output_axes_info.items(), key=lambda item: item[1]["position"]
    )
    for _, info in sorted_output:
        if info["class"] == "ellipsis":
            ell_names = [f"_ell_{i}" for i in range(ellipsis_ndim)]
            output_expanded_names.extend(ell_names)
            try:
                inds = [input_expanded_axis_names.index(name) for name in ell_names]
            except ValueError as e:
                raise EinopsError(
                    f"Ellipsis indices not found in input names: {input_expanded_axis_names}. Error: {e}"
                )
            transpose_order.extend(inds)
            final_target_shape.extend([tensor.shape[i] for i in inds])
        elif info["class"] == "composite":
            comps = info["components"]
            output_expanded_names.extend(comps)
            try:
                inds = [input_expanded_axis_names.index(comp) for comp in comps]
            except ValueError as e:
                raise EinopsError(
                    f"One or more composite components {comps} not found in input names: {input_expanded_axis_names}. Error: {e}"
                )
            transpose_order.extend(inds)
            merged = int(np.prod([resolved_dims[c] for c in comps]))
            final_target_shape.append(merged)
        elif info["class"] == "literal":
            final_target_shape.append(1)
        else:
            token = info["axis_name"]
            output_expanded_names.append(token)
            final_target_shape.append(resolved_dims[token])
            if token in new_output_axes:
                found = False
                for idx in literal_one_indices:
                    if not literal_one_usage[idx]:
                        transpose_order.append(idx)
                        literal_one_usage[idx] = True
                        found = True
                        break
                if not found:
                    raise EinopsError(
                        f"Not enough literal '1's to replace new output axis '{token}'."
                    )
            else:
                try:
                    idx = input_expanded_axis_names.index(token)
                    transpose_order.append(idx)
                except ValueError:
                    raise EinopsError(
                        f"Input axis '{token}' not found among expanded names: {input_expanded_axis_names}."
                    )

    if tensor.ndim != len(input_expanded_axis_names):
        raise EinopsError(
            f"Internal error: expanded tensor shape {tensor.shape} does not match expected dims {input_expanded_axis_names}."
        )
    transposed = tensor.transpose(transpose_order)

    broadcast_shape = []
    temp_idx = 0
    sorted_output = sorted(
        output_axes_info.items(), key=lambda item: item[1]["position"]
    )
    for _, info in sorted_output:
        if info["class"] == "ellipsis":
            shape_slice = list(transposed.shape[temp_idx : temp_idx + ellipsis_ndim])
            broadcast_shape.extend(shape_slice)
            temp_idx += ellipsis_ndim
        elif info["class"] == "composite":
            num = len(info["components"])
            shape_slice = list(transposed.shape[temp_idx : temp_idx + num])
            broadcast_shape.extend(shape_slice)
            temp_idx += num
        elif info["class"] == "literal":
            token = info["axis_name"]
            broadcast_shape.append(resolved_dims.get(token, 1))
        else:
            token = info["axis_name"]
            if token in new_output_axes:
                broadcast_shape.append(resolved_dims[token])
                temp_idx += 1
            else:
                broadcast_shape.append(transposed.shape[temp_idx])
                temp_idx += 1

    if tuple(broadcast_shape) != transposed.shape:
        try:
            transposed = np.broadcast_to(transposed, broadcast_shape)
        except ValueError as e:
            raise EinopsError(
                f"Failed to broadcast tensor from {transposed.shape} to target shape {broadcast_shape}. "
                f"Error: {e}"
            )

    prod_bcast = int(np.prod(transposed.shape))
    prod_target = int(np.prod(final_target_shape))
    if prod_bcast != prod_target:
        raise EinopsError(
            f"Final element count mismatch: broadcasted tensor has product {prod_bcast} vs expected {prod_target} from shape {final_target_shape}."
        )
    try:
        output_tensor = transposed.reshape(final_target_shape)
    except Exception as e:
        raise EinopsError(
            f"Final reshape failed to shape {final_target_shape} from tensor with shape {transposed.shape}. Error: {e}"
        )

    return output_tensor


def rearrange(tensor: np.ndarray, pattern: str, **axes_length) -> np.ndarray:
    """Rearranges dimensions of a NumPy ndarray based on the einops-style pattern.

    Args:
        tensor: The input tensor.
        pattern: The einops pattern string (e.g., "b c h w -> b (h w) c").
        **axes_length: Keyword arguments specifying lengths of named axes
                       (e.g., h=224, w=224). Required for decomposing axes
                       when the dimension size is not unique.

    Raises:
        EinopsError: If the pattern is invalid, dimensions are ambiguous or
                        mismatch the tensor shape.
        TypeError: If the input tensor is not a NumPy ndarray.

    Returns:
        np.ndarray: The rearranged tensor.
    """
    if not isinstance(tensor, np.ndarray):
        raise TypeError("Input tensor must be a NumPy ndarray.")

    try:
        left_axes, right_axes = pattern_parser(pattern)
        check_and_compare_sides(left_axes, right_axes, set(axes_length.keys()))
        input_info = parse_axis(left_axes)
        output_info = parse_axis(right_axes)
        mapper_result = axis_mapper(tensor.shape, input_info, axes_length)
        result_tensor = apply_operation(
            tensor,
            input_info,
            output_info,
            mapper_result,
        )
        return result_tensor
    except (EinopsError, ValueError, TypeError) as e:
        raise e
    except Exception as e:
        raise EinopsError(f"An unexpected internal error occurred: {e}")
