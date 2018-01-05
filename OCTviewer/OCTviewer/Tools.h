#pragma once

using size = size_t;

enum visualizationType { TYPE_2D, TYPE_3D };

struct visualizationParams {

	visualizationType type;
	size xSizeScale;
	size ySizeScale;
	size zSizeScale;
	size ascanSize_px;
	size bscanSize_px;
	size numberOfBscans;
	size y_mm;
	size x_mm;
	size z_mm;

};
