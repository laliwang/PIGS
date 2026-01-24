
#ifndef CORE_MESH_POINTIO_INL_H_
#define CORE_MESH_POINTIO_INL_H_

namespace ml {



	template <class FloatType>
	void PointCloudIO<FloatType>::loadFromPLY( const std::string& filename, PointCloud<FloatType>& pc )
	{
		std::ifstream file(filename, std::ios::binary);
		if (!file.is_open())	throw MLIB_EXCEPTION("Could not open file " + filename);			

		PlyHeader header(file);

		if (header.m_numVertices == (unsigned int)-1) throw MLIB_EXCEPTION("no vertices found");

		pc.m_points.resize(header.m_numVertices);
		if (header.m_bHasNormals)	pc.m_normals.resize(header.m_numVertices);
		if (header.m_bHasColors)	pc.m_colors.resize(header.m_numVertices);

		if (header.m_bBinary) {
			unsigned long long size = 0;
			for (size_t i = 0; i < header.m_properties["vertex"].size(); i++) {
				size += header.m_properties["vertex"][i].byteSize;
			}
			// create an empty array
			char* data = new char[size*header.m_numVertices];
			// fill the array from file
			file.read(data, size*header.m_numVertices);
			// move data to the pointcloud object
			for (unsigned long long i = 0; i < header.m_numVertices; i++) {
				unsigned int byteOffset = 0;
				const std::vector<PlyHeader::PlyPropertyHeader>& vertexProperties = header.m_properties["vertex"];

				for (unsigned int j = 0; j < vertexProperties.size(); j++) {
					char* data_ptr = &data[i * size + byteOffset];

					// handle float/double vals for vertices and normals
					// assume colors are always uchar
					// TODO: handle any data type here dynamically
					float val;

					if (vertexProperties[j].nameType == "float") {
						// get the float
						val = ((float*) data_ptr)[0];
					}
					else if (vertexProperties[j].nameType == "double") {
						// downcast double to float
						val = static_cast<float>(((double*) data_ptr)[0]);
					}
					else if (vertexProperties[j].nameType == "uchar") {
						// get the uchar, divide by 255 to normalize to 0-1 range and store in a float
						val = ((unsigned char*)data_ptr)[0] / 255.0f;
					}

					if (vertexProperties[j].name == "x") {
						pc.m_points[i].x = val;
					}
					else if (vertexProperties[j].name == "y") {
						pc.m_points[i].y = val;
					}
					else if (vertexProperties[j].name == "z") {
						pc.m_points[i].z = val;
					}
					else if (vertexProperties[j].name == "nx") {
						pc.m_normals[i].x = val;
					}
					else if (vertexProperties[j].name == "ny") {
						pc.m_normals[i].y = val;
					}
					else if (vertexProperties[j].name == "nz") {
						pc.m_normals[i].z = val;
					}
					else if (vertexProperties[j].name == "red") {
						pc.m_colors[i].x = val;
					}
					else if (vertexProperties[j].name == "green") {
						pc.m_colors[i].y = val;
					}
					else if (vertexProperties[j].name == "blue") {
						pc.m_colors[i].z = val;
					}
					else if (vertexProperties[j].name == "alpha") {
						pc.m_colors[i].w = val;
					} 
					//unknown property -> ignore
					byteOffset += vertexProperties[j].byteSize;
				}
				assert(byteOffset == size);

			}	

			delete [] data;
		} else {
			MLIB_WARNING("untested");
			for (size_t i = 0; i < header.m_numVertices; i++) {
				std::string line;
				std::getline(file, line);
				std::stringstream ss(line); // TODO replace with sscanf which should be faster
				ss >> pc.m_points[i].x >> pc.m_points[i].y >> pc.m_points[i].z;
				if (header.m_bHasColors) {
					ss >> pc.m_colors[i].x >> pc.m_colors[i].y >> pc.m_colors[i].z;
					pc.m_colors[i] /= (FloatType)255.0;
				}
			}
		}

		file.close();
	}

} // namespace ml

#endif