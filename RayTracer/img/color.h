#pragma once

#include <iostream>
#include <Eigen/Dense>

namespace rayUtilities {
	using Color = Eigen::Matrix<double, 3, 1>;

	void write_color(std::ofstream& out, Color color, const int nSample) {
		const auto pixelColor = (color / (double)nSample).cwiseSqrt().cwiseMin(.999).cwiseMax(0.);

		out << static_cast<int> (256 * pixelColor[0]) <<" "<<
			static_cast<int> (256 * pixelColor[1]) << " " <<
			static_cast<int> (256 * pixelColor[2]) << std::endl;
	}
}