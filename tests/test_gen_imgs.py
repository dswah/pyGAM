from unittest.mock import patch

# Import the function to test
import gen_imgs


def test_gen_basis_fns():
    """
    Test that gen_basis_fns function executes without error and calls plt.savefig
    with the correct filename and dpi parameters for basis function visualization.
    """
    with patch("gen_imgs.plt.savefig") as mock_savefig:
        gen_imgs.gen_basis_fns()
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert args[0] == "imgs/pygam_basis.png"
        assert kwargs["dpi"] == 300


def test_cake_data_in_one():
    """
    Test that cake_data_in_one function executes without error and calls plt.savefig
    with the correct filename and dpi parameters for cake dataset visualization.
    """
    with patch("gen_imgs.plt.savefig") as mock_savefig:
        gen_imgs.cake_data_in_one()
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert args[0] == "imgs/pygam_cake_data.png"
        assert kwargs["dpi"] == 300


def test_faithful_data_poisson():
    """
    Test that faithful_data_poisson function executes without error and calls
    plt.savefig with the correct filename and dpi parameters for Poisson GAM
    visualization.
    """
    with patch("gen_imgs.plt.savefig") as mock_savefig:
        gen_imgs.faithful_data_poisson()
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert args[0] == "imgs/pygam_poisson.png"
        assert kwargs["dpi"] == 300


def test_single_data_linear():
    """
    Test that single_data_linear function executes without error and calls plt.savefig
    with the correct filename and dpi parameters for single variable linear GAM.
    """
    with patch("gen_imgs.plt.savefig") as mock_savefig:
        gen_imgs.single_data_linear()
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert args[0] == "imgs/pygam_single_pred_linear.png"
        assert kwargs["dpi"] == 300


def test_mcycle_data_linear():
    """
    Test that mcycle_data_linear function executes without error and calls plt.savefig
    twice with the correct filenames and dpi parameters for motorcycle data plots.
    """
    with patch("gen_imgs.plt.savefig") as mock_savefig:
        gen_imgs.mcycle_data_linear()
        assert mock_savefig.call_count == 2
        args1, kwargs1 = mock_savefig.call_args_list[0]
        args2, kwargs2 = mock_savefig.call_args_list[1]
        assert args1[0] == "imgs/pygam_mcycle_data_linear.png"
        assert kwargs1["dpi"] == 300
        assert args2[0] == "imgs/pygam_mcycle_data_extrapolation.png"
        assert kwargs2["dpi"] == 300


def test_wage_data_linear():
    """
    Test that wage_data_linear function executes without error and calls plt.savefig
    with the correct filename and dpi parameters for wage dataset visualization.
    """
    with patch("gen_imgs.plt.savefig") as mock_savefig:
        gen_imgs.wage_data_linear()
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert args[0] == "imgs/pygam_wage_data_linear.png"
        assert kwargs["dpi"] == 300


def test_default_data_logistic():
    """
    Test that default_data_logistic function executes without error and calls
    plt.savefig with the correct filename and dpi parameters for logistic GAM
    visualization.
    """
    with patch("gen_imgs.plt.savefig") as mock_savefig:
        gen_imgs.default_data_logistic()
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert args[0] == "imgs/pygam_default_data_logistic.png"
        assert kwargs["dpi"] == 300


def test_constraints():
    """
    Test that constraints function executes without error and calls plt.savefig
    with the correct filename and dpi parameters for constraint visualization.
    """
    with patch("gen_imgs.plt.savefig") as mock_savefig:
        gen_imgs.constraints()
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert args[0] == "imgs/pygam_constraints.png"
        assert kwargs["dpi"] == 300


def test_trees_data_custom():
    """
    Test that trees_data_custom function executes without error and calls plt.savefig
    with the correct filename and dpi parameters for custom GAM visualization.
    """
    with patch("gen_imgs.plt.savefig") as mock_savefig:
        gen_imgs.trees_data_custom()
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert args[0] == "imgs/pygam_custom.png"
        assert kwargs["dpi"] == 300


def test_gen_multi_data():
    """
    Test that gen_multi_data function executes without error and calls plt.savefig
    twice with the correct filenames and dpi parameters for multivariate data plots.
    """
    with patch("gen_imgs.plt.savefig") as mock_savefig:
        gen_imgs.gen_multi_data()
        assert mock_savefig.call_count == 2
        args1, kwargs1 = mock_savefig.call_args_list[0]
        args2, kwargs2 = mock_savefig.call_args_list[1]
        assert args1[0] == "imgs/pygam_multi_pdep.png"
        assert kwargs1["dpi"] == 300
        assert args2[0] == "imgs/pygam_multi_deviance.png"
        assert kwargs2["dpi"] == 300


def test_gen_tensor_data():
    """
    Test that gen_tensor_data function executes without error and calls plt.savefig
    with the correct filename and dpi parameters for tensor product visualization.
    """
    with patch("gen_imgs.plt.savefig") as mock_savefig:
        gen_imgs.gen_tensor_data()
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert args[0] == "imgs/pygam_tensor.png"
        assert kwargs["dpi"] == 300


def test_chicago_tensor():
    """
    Test that chicago_tensor function executes without error and calls
    plt.savefig with the correct filename and dpi parameters for Chicago
    dataset tensor visualization.
    """
    with patch("gen_imgs.plt.savefig") as mock_savefig:
        gen_imgs.chicago_tensor()
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert args[0] == "imgs/pygam_chicago_tensor.png"
        assert kwargs["dpi"] == 300


def test_expectiles():
    """
    Test that expectiles function executes without error and calls plt.savefig
    with the correct filename and dpi parameters for expectile GAM visualization.
    """
    with patch("gen_imgs.plt.savefig") as mock_savefig:
        gen_imgs.expectiles()
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert args[0] == "imgs/pygam_expectiles.png"
        assert kwargs["dpi"] == 300
