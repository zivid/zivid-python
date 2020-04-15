#include <ZividPython/NodeType.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapEnum(pybind11::enum_<Zivid::DataModel::NodeType> pyEnum)
    {
        pyEnum.value("group", Zivid::DataModel::NodeType::group)
            .value("leaf_data_model_list", Zivid::DataModel::NodeType::leafDataModelList)
            .value("leaf_value", Zivid::DataModel::NodeType::leafValue)
            .export_values();
    }
} // namespace ZividPython