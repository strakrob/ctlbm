#include "lbm.cuh"

#include <fstream>
#include <iomanip>
#include <sstream>

namespace lbm {

namespace {

void write_vti_image(
    const std::string& filename,
    int x0,
    int x1,
    int y0,
    int y1,
    int z0,
    int z1,
    int origin_x,
    int origin_y,
    int origin_z,
    const std::vector<Real>& rho,
    const std::vector<Real>& ux,
    const std::vector<Real>& uy,
    const std::vector<Real>& uz) {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Unable to open VTI output file: " + filename);
    }

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <ImageData WholeExtent=\"" << x0 << ' ' << x1 << ' ' << y0 << ' ' << y1 << ' ' << z0 << ' ' << z1
        << "\" Origin=\"" << origin_x << ' ' << origin_y << ' ' << origin_z << "\" Spacing=\"1 1 1\">\n";
    out << "    <Piece Extent=\"" << x0 << ' ' << x1 << ' ' << y0 << ' ' << y1 << ' ' << z0 << ' ' << z1 << "\">\n";
    out << "      <PointData Scalars=\"rho\" Vectors=\"velocity\">\n";

    out << "        <DataArray type=\"" << kVtiScalarType << "\" Name=\"rho\" format=\"ascii\">\n          ";
    for (std::size_t i = 0; i < rho.size(); ++i) {
        out << std::setprecision(16) << rho[i] << ' ';
        if ((i + 1) % 6 == 0) {
            out << "\n          ";
        }
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"" << kVtiScalarType << "\" Name=\"ux\" format=\"ascii\">\n          ";
    for (std::size_t i = 0; i < ux.size(); ++i) {
        out << std::setprecision(16) << ux[i] << ' ';
        if ((i + 1) % 6 == 0) {
            out << "\n          ";
        }
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"" << kVtiScalarType << "\" Name=\"uy\" format=\"ascii\">\n          ";
    for (std::size_t i = 0; i < uy.size(); ++i) {
        out << std::setprecision(16) << uy[i] << ' ';
        if ((i + 1) % 6 == 0) {
            out << "\n          ";
        }
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"" << kVtiScalarType << "\" Name=\"uz\" format=\"ascii\">\n          ";
    for (std::size_t i = 0; i < uz.size(); ++i) {
        out << std::setprecision(16) << uz[i] << ' ';
        if ((i + 1) % 6 == 0) {
            out << "\n          ";
        }
    }
    out << "\n        </DataArray>\n";

    out << "        <DataArray type=\"" << kVtiScalarType << "\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n          ";
    for (std::size_t i = 0; i < ux.size(); ++i) {
        out << std::setprecision(16) << ux[i] << ' ' << uy[i] << ' ' << uz[i] << ' ';
        if ((i + 1) % 2 == 0) {
            out << "\n          ";
        }
    }
    out << "\n        </DataArray>\n";

    out << "      </PointData>\n";
    out << "      <CellData/>\n";
    out << "    </Piece>\n";
    out << "  </ImageData>\n";
    out << "</VTKFile>\n";
}

}  // namespace

void write_vti(
    const std::string& filename,
    const SimulationConfig& cfg,
    const std::vector<Real>& rho,
    const std::vector<Real>& ux,
    const std::vector<Real>& uy,
    const std::vector<Real>& uz) {
    write_vti_image(filename, 0, cfg.nx - 1, 0, cfg.ny - 1, 0, cfg.nz - 1, 0, 0, 0, rho, ux, uy, uz);
}

void write_vti_midplane_cross_sections(
    const std::string& output_stem,
    const SimulationConfig& cfg,
    const std::vector<Real>& rho,
    const std::vector<Real>& ux,
    const std::vector<Real>& uy,
    const std::vector<Real>& uz) {
    const int mid_x = cfg.nx / 2;
    const int mid_y = cfg.ny / 2;
    const int mid_z = cfg.nz / 2;

    std::vector<Real> rho_xy;
    std::vector<Real> ux_xy;
    std::vector<Real> uy_xy;
    std::vector<Real> uz_xy;
    rho_xy.reserve(static_cast<std::size_t>(cfg.nx) * cfg.ny);
    ux_xy.reserve(static_cast<std::size_t>(cfg.nx) * cfg.ny);
    uy_xy.reserve(static_cast<std::size_t>(cfg.nx) * cfg.ny);
    uz_xy.reserve(static_cast<std::size_t>(cfg.nx) * cfg.ny);
    for (int y = 0; y < cfg.ny; ++y) {
        for (int x = 0; x < cfg.nx; ++x) {
            const int cell = flatten_xyz(x, y, mid_z, cfg.nx, cfg.ny, cfg.nz);
            rho_xy.push_back(rho[cell]);
            ux_xy.push_back(ux[cell]);
            uy_xy.push_back(uy[cell]);
            uz_xy.push_back(uz[cell]);
        }
    }
    write_vti_image(output_stem + "_xy.vti", 0, cfg.nx - 1, 0, cfg.ny - 1, 0, 0, 0, 0, mid_z, rho_xy, ux_xy, uy_xy, uz_xy);

    std::vector<Real> rho_xz;
    std::vector<Real> ux_xz;
    std::vector<Real> uy_xz;
    std::vector<Real> uz_xz;
    rho_xz.reserve(static_cast<std::size_t>(cfg.nx) * cfg.nz);
    ux_xz.reserve(static_cast<std::size_t>(cfg.nx) * cfg.nz);
    uy_xz.reserve(static_cast<std::size_t>(cfg.nx) * cfg.nz);
    uz_xz.reserve(static_cast<std::size_t>(cfg.nx) * cfg.nz);
    for (int z = 0; z < cfg.nz; ++z) {
        for (int x = 0; x < cfg.nx; ++x) {
            const int cell = flatten_xyz(x, mid_y, z, cfg.nx, cfg.ny, cfg.nz);
            rho_xz.push_back(rho[cell]);
            ux_xz.push_back(ux[cell]);
            uy_xz.push_back(uy[cell]);
            uz_xz.push_back(uz[cell]);
        }
    }
    write_vti_image(output_stem + "_xz.vti", 0, cfg.nx - 1, 0, 0, 0, cfg.nz - 1, 0, mid_y, 0, rho_xz, ux_xz, uy_xz, uz_xz);

    std::vector<Real> rho_yz;
    std::vector<Real> ux_yz;
    std::vector<Real> uy_yz;
    std::vector<Real> uz_yz;
    rho_yz.reserve(static_cast<std::size_t>(cfg.ny) * cfg.nz);
    ux_yz.reserve(static_cast<std::size_t>(cfg.ny) * cfg.nz);
    uy_yz.reserve(static_cast<std::size_t>(cfg.ny) * cfg.nz);
    uz_yz.reserve(static_cast<std::size_t>(cfg.ny) * cfg.nz);
    for (int z = 0; z < cfg.nz; ++z) {
        for (int y = 0; y < cfg.ny; ++y) {
            const int cell = flatten_xyz(mid_x, y, z, cfg.nx, cfg.ny, cfg.nz);
            rho_yz.push_back(rho[cell]);
            ux_yz.push_back(ux[cell]);
            uy_yz.push_back(uy[cell]);
            uz_yz.push_back(uz[cell]);
        }
    }
    write_vti_image(output_stem + "_yz.vti", 0, 0, 0, cfg.ny - 1, 0, cfg.nz - 1, mid_x, 0, 0, rho_yz, ux_yz, uy_yz, uz_yz);
}

}  // namespace lbm
