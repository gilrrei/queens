"""TODO_doc."""

import os

import numpy as np
import pytest

from pqueens.external_geometry.baci_dat_geometry import BaciDatExternalGeometry


# general input fixtures
@pytest.fixture()
def default_geo_obj(tmpdir):
    """TODO_doc."""
    path_to_dat_file = tmpdir.join('myfile.dat')
    list_geometric_sets = ["DSURFACE 9"]
    list_associated_material_numbers = [[10, 11]]
    element_topology = [{"element_number": [], "nodes": [], "material": []}]
    node_topology = [{"node_mesh": [], "node_topology": [], "topology_name": ""}]
    line_topology = [{"node_mesh": [], "line_topology": [], "topology_name": ""}]
    surface_topology = [{"node_mesh": [], "surface_topology": [], "topology_name": ""}]
    volume_topology = [{"node_mesh": [], "volume_topology": [], "topology_name": ""}]
    node_coordinates = {"node_mesh": [], "coordinates": []}

    path_to_preprocessed_dat_file = tmpdir.join('preprocessed')
    random_fields = (
        [{"name": "mat_param", "type": "material", "external_instance": "DSURFACE 1"}],
    )

    geo_obj = BaciDatExternalGeometry(
        path_to_dat_file,
        list_geometric_sets,
        list_associated_material_numbers,
        element_topology,
        node_topology,
        line_topology,
        surface_topology,
        volume_topology,
        node_coordinates,
        path_to_preprocessed_dat_file,
        random_fields,
    )
    return geo_obj


def write_to_file(data, filepath):
    """TODO_doc."""
    with open(filepath, 'w') as fp:
        fp.write(data)


@pytest.fixture()
def dat_dummy_comment():
    """TODO_doc."""
    data = [
        '// this is a comment\n',
        ' // this is another comment --------------------------------------'
        '-----------------NODE COORDS',
    ]
    data = ''.join(data)
    return data


@pytest.fixture()
def dat_dummy_get_fun():
    """TODO_doc."""
    data = [
        'NODE    3419 DSURFACE 10\n',
        'NODE    3421 DSURFACE 10\n',
        'NODE    3423 DSURFACE 10\n',
        'NODE    3425 DSURFACE 10',
    ]
    data = ''.join(data)
    return data


@pytest.fixture(
    params=[
        '------------------------------------------------DESIGN DESCRIPTION    ',
        '------------------------------------------------DNODE-NODE TOPOLOGY   ',
        '------------------------------------------------DLINE-NODE TOPOLOGY   ',
        '------------------------------------------------DSURF-NODE TOPOLOGY   ',
        '------------------------------------------------DVOL-NODE TOPOLOGY    ',
        '------------------------------------------------NODE COORDS//         ',
    ]
)
def dat_section_true(request):
    """TODO_doc."""
    return request.param


@pytest.fixture(
    params=[
        '//------------------------------------------------DESIGN DESCRIPTION    ',
        ' // ------------------------------------------------DNODE-NODE TOPOLOGY   ',
        '------- //-----------------------------------------DLINE-NODE TOPOLOGY   ',
        '------------------------------------------------DSRF-NDE TOOGY   ',
        '------------------------------------------------VOL-NODE TOPOLOGY    ',
        '------------------------------------------------NODECOORDS//           ',
    ]
)
def dat_section_false(request):
    """TODO_doc."""
    return request.param


@pytest.fixture(
    params=[
        'DNODE-NODE TOPOLOGY',
        'DLINE-NODE TOPOLOGY',
        'DSURF-NODE TOPOLOGY',
        'DVOL-NODE TOPOLOGY',
    ]
)
def current_dat_sections(request):
    """TODO_doc."""
    return request.param


@pytest.fixture()
def desired_sections():
    """TODO_doc."""
    sections = {
        'DLINE-NODE TOPOLOGY': ['DLINE 1'],
        'DSURF-NODE TOPOLOGY': ['DSURFACE 2', 'DSURFACE 1'],
        'DNODE-NODE TOPOLOGY': ['DNODE 1'],
        'DVOL-NODE TOPOLOGY': ['DVOL 1'],
    }
    return sections


@pytest.fixture()
def default_coords():
    """TODO_doc."""
    coords = [
        'NODE 1 COORD -1.0000000000000000e+00 -2.5000000000000000e-01 0.0000000000000000e+00',
        'NODE 2 COORD -1.0000000000000000e+00 -2.5000000000000000e-01 -3.7500000000000006e-02',
        'NODE 3 COORD -1.0000000000000000e+00 -2.1666666666666665e-01 -3.7500000000000006e-02',
        'NODE 4 COORD -1.0000000000000000e+00 -2.1666666666666662e-01 0.0000000000000000e+00',
    ]
    return coords


@pytest.fixture()
def default_topology_node():
    """TODO_doc."""
    data = [
        'NODE    1 DNODE 1',
        'NODE    2 DNODE 1',
        'NODE    4 DNODE 1',
        'NODE    9 DNODE 1',
        'NODE    13 DNODE 1',
    ]
    return data


@pytest.fixture()
def default_topology_line():
    """TODO_doc."""
    data = [
        'NODE    1 DLINE 1',
        'NODE    2 DLINE 1',
        'NODE    4 DLINE 1',
        'NODE    9 DLINE 1',
        'NODE    13 DLINE 1',
    ]
    return data


@pytest.fixture()
def default_topology_surf():
    """TODO_doc."""
    data = [
        'NODE    1 DSURFACE 1',
        'NODE    2 DSURFACE 1',
        'NODE    4 DSURFACE 1',
        'NODE    9 DSURFACE 1',
        'NODE    13 DSURFACE 1',
    ]
    return data


@pytest.fixture()
def default_topology_vol():
    """TODO_doc."""
    data = [
        'NODE    1 DVOL 1',
        'NODE    2 DVOL 1',
        'NODE    4 DVOL 1',
        'NODE    9 DVOL 1',
        'NODE    13 DVOL 1',
    ]
    return data


# ----------------- actual unit_tests -------------------------------------------------------------
def test_init(mocker, tmpdir):
    """TODO_doc."""
    path_to_dat_file = 'dummy_path'
    list_geometric_sets = ["DSURFACE 9"]
    list_associated_material_numbers = [[10, 11]]
    element_topology = [{"element_number": [], "nodes": [], "material": []}]
    node_topology = [{"node_mesh": [], "node_topology": [], "topology_name": ""}]
    line_topology = [{"node_mesh": [], "line_topology": [], "topology_name": ""}]
    surface_topology = [{"node_mesh": [], "line_topology": [], "topology_name": ""}]
    volume_topology = [{"node_mesh": [], "volume_topology": [], "topology_name": ""}]
    node_coordinates = {"node_mesh": [], "coordinates": []}
    mp = mocker.patch('pqueens.external_geometry.external_geometry.ExternalGeometry.__init__')

    path_to_preprocessed_dat_file = tmpdir.join('preprocessed')
    random_fields = (
        [{"name": "mat_param", "type": "material", "external_instance": "DSURFACE 1"}],
    )

    geo_obj = BaciDatExternalGeometry(
        path_to_dat_file,
        list_geometric_sets,
        list_associated_material_numbers,
        element_topology,
        node_topology,
        line_topology,
        surface_topology,
        volume_topology,
        node_coordinates,
        path_to_preprocessed_dat_file,
        random_fields,
    )
    mp.assert_called_once()
    assert geo_obj.path_to_dat_file == path_to_dat_file
    assert geo_obj.list_geometric_sets == list_geometric_sets
    assert geo_obj.current_dat_section is None
    assert geo_obj.current_dat_section is None
    assert geo_obj.design_description == {}
    assert geo_obj.node_topology == node_topology
    assert geo_obj.surface_topology == surface_topology
    assert geo_obj.volume_topology == volume_topology
    assert geo_obj.desired_dat_sections == {}
    assert geo_obj.nodes_of_interest is None
    assert geo_obj.node_coordinates == node_coordinates
    assert geo_obj.path_to_preprocessed_dat_file == path_to_preprocessed_dat_file
    assert geo_obj.random_fields == random_fields


def test_read_external_data_comment(mocker, tmpdir, dat_dummy_comment, default_geo_obj):
    """TODO_doc."""
    filepath = tmpdir.join("myfile.dat")
    write_to_file(dat_dummy_comment, filepath)

    mocker.patch(
        'pqueens.external_geometry.baci_dat_geometry.BaciDatExternalGeometry'
        '._get_current_dat_section',
        return_value=False,
    )

    mocker.patch(
        'pqueens.external_geometry.baci_dat_geometry.BaciDatExternalGeometry'
        '._get_design_description',
        return_value=1,
    )

    mocker.patch(
        'pqueens.external_geometry.baci_dat_geometry.BaciDatExternalGeometry'
        '._get_only_desired_topology',
        return_value=1,
    )

    mocker.patch(
        'pqueens.external_geometry.baci_dat_geometry.BaciDatExternalGeometry'
        '._get_only_desired_coordinates',
        return_value=1,
    )

    mocker.patch(
        'pqueens.external_geometry.baci_dat_geometry.BaciDatExternalGeometry._get_materials',
        return_value=1,
    )

    mocker.patch(
        'pqueens.external_geometry.baci_dat_geometry.BaciDatExternalGeometry'
        '._get_elements_belonging_to_desired_material',
        return_value=1,
    )

    default_geo_obj._read_geometry_from_dat_file()

    assert default_geo_obj._get_current_dat_section.call_count == 2
    assert default_geo_obj._get_design_description.call_count == 0
    assert default_geo_obj._get_only_desired_topology.call_count == 0
    assert default_geo_obj._get_only_desired_coordinates.call_count == 0
    assert default_geo_obj._get_materials.call_count == 0
    assert default_geo_obj._get_elements_belonging_to_desired_material.call_count == 0


def test_read_external_data_get_functions(mocker, tmpdir, dat_dummy_get_fun, default_geo_obj):
    """TODO_doc."""
    filepath = tmpdir.join("myfile.dat")
    write_to_file(dat_dummy_get_fun, filepath)

    default_geo_obj.current_dat_section = 'dummy'

    mocker.patch(
        'pqueens.external_geometry.baci_dat_geometry.BaciDatExternalGeometry'
        '._get_current_dat_section',
        return_value=False,
    )

    mocker.patch(
        'pqueens.external_geometry.baci_dat_geometry.BaciDatExternalGeometry'
        '._get_design_description',
        return_value=1,
    )

    mocker.patch(
        'pqueens.external_geometry.baci_dat_geometry.BaciDatExternalGeometry'
        '._get_only_desired_topology',
        return_value=1,
    )

    mocker.patch(
        'pqueens.external_geometry.baci_dat_geometry.BaciDatExternalGeometry'
        '._get_only_desired_coordinates',
        return_value=1,
    )

    mocker.patch(
        'pqueens.external_geometry.baci_dat_geometry.BaciDatExternalGeometry' '._get_materials',
        return_value=1,
    )

    mocker.patch(
        'pqueens.external_geometry.baci_dat_geometry.BaciDatExternalGeometry'
        '._get_elements_belonging_to_desired_material',
        return_value=1,
    )

    default_geo_obj._read_geometry_from_dat_file()

    assert default_geo_obj._get_current_dat_section.call_count == 4
    assert default_geo_obj._get_design_description.call_count == 4
    assert default_geo_obj._get_only_desired_topology.call_count == 4
    assert default_geo_obj._get_only_desired_coordinates.call_count == 4
    assert default_geo_obj._get_materials.call_count == 4
    assert default_geo_obj._get_elements_belonging_to_desired_material.call_count == 4


def test_organize_sections(default_geo_obj):
    """Wrapper for *_get_desired_dat_sections*."""
    desired_geo_sets = ['DSURFACE 9', 'DVOL 2', 'DLINE 1', 'DSURFACE 8']
    expected_dat_section = {
        'DLINE-NODE TOPOLOGY': ['DLINE 1'],
        'DSURF-NODE TOPOLOGY': ['DSURFACE 9', 'DSURFACE 8'],
        'DVOL-NODE TOPOLOGY': ['DVOL 2'],
    }
    default_geo_obj.list_geometric_sets = desired_geo_sets
    default_geo_obj._get_desired_dat_sections()

    assert default_geo_obj.desired_dat_sections == expected_dat_section


def test_get_current_dat_section_true(default_geo_obj, dat_section_true):
    """TODO_doc."""
    default_geo_obj._get_current_dat_section(dat_section_true)
    clean_section_name = dat_section_true.strip()
    clean_section_name = clean_section_name.strip('-')
    assert default_geo_obj.current_dat_section == clean_section_name


def test_get_current_dat_section_false(default_geo_obj, dat_section_false):
    """TODO_doc."""
    default_geo_obj._get_current_dat_section(dat_section_false)
    assert default_geo_obj.current_dat_section is None


def test_check_if_in_desired_dat_section(default_geo_obj):
    """TODO_doc."""
    default_geo_obj.desired_dat_sections = {
        'DLINE-NODE TOPOLOGY': ['DLINE 1'],
        'DSURF-NODE TOPOLOGY': ['DSURFACE 9', 'DSURFACE 8'],
    }

    # return true
    default_geo_obj.current_dat_section = 'DSURF-NODE TOPOLOGY'
    return_value = default_geo_obj._check_if_in_desired_dat_section()
    assert return_value

    # return false
    default_geo_obj.current_dat_section = 'DVOL-NODE TOPOLOGY'
    return_value = default_geo_obj._check_if_in_desired_dat_section()
    assert not return_value


def test_get_topology(
    tmpdir,
    default_geo_obj,
    current_dat_sections,
    desired_sections,
    default_topology_node,
    default_topology_line,
    default_topology_surf,
    default_topology_vol,
):
    """TODO_doc."""
    default_geo_obj.current_dat_section = current_dat_sections
    default_geo_obj.desired_dat_sections = desired_sections

    if current_dat_sections == 'DNODE-NODE TOPOLOGY':
        for line in default_topology_node:
            default_geo_obj._get_topology(line)
        assert default_geo_obj.node_topology[0]['node_mesh'] == [1, 2, 4, 9, 13]
        assert default_geo_obj.node_topology[0]['node_topology'] == [1, 1, 1, 1, 1]

    elif current_dat_sections == 'DLINE-NODE TOPOLOGY':
        for line in default_topology_line:
            default_geo_obj._get_topology(line)
        assert default_geo_obj.line_topology[0]['node_mesh'] == [1, 2, 4, 9, 13]
        assert default_geo_obj.line_topology[0]['line_topology'] == [1, 1, 1, 1, 1]

    elif current_dat_sections == 'DSURF-NODE TOPOLOGY':
        for line in default_topology_surf:
            default_geo_obj._get_topology(line)
        assert default_geo_obj.surface_topology[0]['node_mesh'] == [1, 2, 4, 9, 13]
        assert default_geo_obj.surface_topology[0]['surface_topology'] == [1, 1, 1, 1, 1]

    elif current_dat_sections == 'DVOL-NODE TOPOLOGY':
        for line in default_topology_vol:
            default_geo_obj._get_topology(line)
        assert default_geo_obj.volume_topology[0]['node_mesh'] == [1, 2, 4, 9, 13]
        assert default_geo_obj.volume_topology[0]['volume_topology'] == [1, 1, 1, 1, 1]


def test_get_only_desired_topology(default_geo_obj, mocker):
    """TODO_doc."""
    line = 'dummy'
    mocker.patch(
        'pqueens.external_geometry.baci_dat_geometry.BaciDatExternalGeometry'
        '._check_if_in_desired_dat_section',
        return_value=True,
    )
    mp1 = mocker.patch(
        'pqueens.external_geometry.baci_dat_geometry.BaciDatExternalGeometry' '._get_topology',
        return_value=None,
    )
    default_geo_obj._get_only_desired_topology(line)
    assert mp1.call_count == 1

    mocker.patch(
        'pqueens.external_geometry.baci_dat_geometry.BaciDatExternalGeometry'
        '._check_if_in_desired_dat_section',
        return_value=False,
    )
    # counter stays 1 as called before
    default_geo_obj._get_only_desired_topology(line)
    assert mp1.call_count == 1


def test_get_only_desired_coordinates(default_geo_obj, mocker):
    """TODO_doc."""
    line = 'dummy 2'
    mp1 = mocker.patch(
        'pqueens.external_geometry.baci_dat_geometry.BaciDatExternalGeometry'
        '._get_nodes_of_interest',
        return_value=[1, 2, 3, 4, 5],
    )
    mp2 = mocker.patch(
        'pqueens.external_geometry.baci_dat_geometry.BaciDatExternalGeometry'
        '._get_coordinates_of_desired_geometric_sets',
        return_value=None,
    )
    default_geo_obj.current_dat_section = 'NODE COORDS'
    default_geo_obj._get_only_desired_coordinates(line)
    mp1.assert_called_once()

    nodes_of_interest = [1, 2, 3, 4, 5]
    default_geo_obj.nodes_of_interest = nodes_of_interest
    default_geo_obj._get_only_desired_coordinates(line)
    mp1.assert_called_once()
    mp2.assert_called_once()


def test_get_coordinates_of_desired_geometric_sets(default_geo_obj, default_coords):
    """TODO_doc."""
    for line in default_coords:
        node_list = line.split()
        default_geo_obj._get_coordinates_of_desired_geometric_sets(node_list)
    assert default_geo_obj.node_coordinates['node_mesh'] == [1, 2, 3, 4]
    np.testing.assert_array_equal(
        default_geo_obj.node_coordinates['coordinates'],
        np.array(
            [
                [-1.0000000000000000e00, -2.5000000000000000e-01, 0.0000000000000000e00],
                [-1.0000000000000000e00, -2.5000000000000000e-01, -3.7500000000000006e-02],
                [-1.0000000000000000e00, -2.1666666666666665e-01, -3.7500000000000006e-02],
                [-1.0000000000000000e00, -2.1666666666666662e-01, 0.0000000000000000e00],
            ]
        ),
    )


def test_get_nodes_of_interest(default_geo_obj):
    """TODO_doc."""
    default_geo_obj.node_topology[-1]['node_mesh'] = [1, 2, 4, 9, 13]
    default_geo_obj.node_topology[-1]['node_topology'] = [1, 1, 1, 1, 1]
    default_geo_obj.line_topology[-1]['node_mesh'] = [1, 2, 4, 9, 13]
    default_geo_obj.line_topology[-1]['line_topology'] = [1, 1, 1, 1, 1]
    default_geo_obj.surface_topology[-1]['node_mesh'] = [1, 2, 4, 9, 13]
    default_geo_obj.surface_topology[-1]['surface_topology'] = [1, 1, 1, 1, 1]
    default_geo_obj.volume_topology[-1]['node_mesh'] = [1, 2, 4, 9, 13]
    default_geo_obj.volume_topology[-1]['volume_topology'] = [1, 1, 1, 1, 1]

    default_geo_obj._get_nodes_of_interest()
    assert default_geo_obj.nodes_of_interest == [1, 2, 4, 9, 13]
