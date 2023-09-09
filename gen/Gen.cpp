#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

constexpr const char *fileHeader = R"Py(
# Autogenerated by mlir-tblgen; don't manually edit.

from ._ods_common import _cext as _ods_cext
from ._ods_common import extend_opview_class as _ods_extend_opview_class, segmented_accessor as _ods_segmented_accessor, equally_sized_accessor as _ods_equally_sized_accessor, get_default_loc_context as _ods_get_default_loc_context, get_op_result_or_value as _get_op_result_or_value, get_op_results_or_values as _get_op_results_or_values
_ods_ir = _ods_cext.ir

try:
  from . import _{0}_ops_ext as _ods_ext_module
except ImportError:
  _ods_ext_module = None

import builtins

)Py";

constexpr const char *dialectClassTemplate = R"Py(
@_ods_cext.register_dialect
class _Dialect(_ods_ir.Dialect):
  DIALECT_NAMESPACE = "{0}"
  pass

)Py";

constexpr const char *dialectExtensionTemplate = R"Py(
from ._{0}_ops_gen import _Dialect
)Py";

constexpr const char *opClassTemplate = R"Py(
@_ods_cext.register_operation(_Dialect)
@_ods_extend_opview_class(_ods_ext_module)
class {0}(_ods_ir.OpView):
  OPERATION_NAME = "{1}"
)Py";

constexpr const char *opClassSizedSegmentsTemplate = R"Py(
  _ODS_{0}_SEGMENTS = {1}
)Py";

constexpr const char *opClassRegionSpecTemplate = R"Py(
  _ODS_REGIONS = ({0}, {1})
)Py";

constexpr const char *opSingleTemplate = R"Py(
  @builtins.property
  def {0}(self):
    return self.operation.{1}s[{2}]
)Py";

constexpr const char *opSingleAfterVariableTemplate = R"Py(
  @builtins.property
  def {0}(self):
    _ods_variadic_group_length = len(self.operation.{1}s) - {2} + 1
    return self.operation.{1}s[{3} + _ods_variadic_group_length - 1]
)Py";

constexpr const char *opOneOptionalTemplate = R"Py(
  @builtins.property
  def {0}(self):
    return None if len(self.operation.{1}s) < {2} else self.operation.{1}s[{3}]
)Py";

constexpr const char *opOneVariadicTemplate = R"Py(
  @builtins.property
  def {0}(self):
    _ods_variadic_group_length = len(self.operation.{1}s) - {2} + 1
    return self.operation.{1}s[{3}:{3} + _ods_variadic_group_length]
)Py";

constexpr const char *opVariadicEqualPrefixTemplate = R"Py(
  @builtins.property
  def {0}(self):
    start, pg = _ods_equally_sized_accessor(operation.{1}s, {2}, {3}, {4}))Py";

constexpr const char *opVariadicEqualSimpleTemplate = R"Py(
    return self.operation.{0}s[start]
)Py";

constexpr const char *opVariadicEqualVariadicTemplate = R"Py(
    return self.operation.{0}s[start:start + pg]
)Py";

constexpr const char *opVariadicSegmentTemplate = R"Py(
  @builtins.property
  def {0}(self):
    {1}_range = _ods_segmented_accessor(
         self.operation.{1}s,
         self.operation.attributes["{1}SegmentSizes"], {2})
    return {1}_range{3}
)Py";

constexpr const char *opVariadicSegmentOptionalTrailingTemplate =
        R"Py([0] if len({0}_range) > 0 else None)Py";

constexpr const char *attributeGetterTemplate = R"Py(
  @builtins.property
  def {0}(self):
    return self.operation.attributes["{1}"]
)Py";

constexpr const char *optionalAttributeGetterTemplate = R"Py(
  @builtins.property
  def {0}(self):
    if "{1}" not in self.operation.attributes:
      return None
    return self.operation.attributes["{1}"]
)Py";

constexpr const char *unitAttributeGetterTemplate = R"Py(
  @builtins.property
  def {0}(self):
    return "{1}" in self.operation.attributes
)Py";

constexpr const char *attributeSetterTemplate = R"Py(
  @{0}.setter
  def {0}(self, value):
    if value is None:
      raise ValueError("'None' not allowed as value for mandatory attributes")
    self.operation.attributes["{1}"] = value
)Py";

constexpr const char *optionalAttributeSetterTemplate = R"Py(
  @{0}.setter
  def {0}(self, value):
    if value is not None:
      self.operation.attributes["{1}"] = value
    elif "{1}" in self.operation.attributes:
      del self.operation.attributes["{1}"]
)Py";

constexpr const char *unitAttributeSetterTemplate = R"Py(
  @{0}.setter
  def {0}(self, value):
    if bool(value):
      self.operation.attributes["{1}"] = _ods_ir.UnitAttr.get()
    elif "{1}" in self.operation.attributes:
      del self.operation.attributes["{1}"]
)Py";

constexpr const char *attributeDeleterTemplate = R"Py(
  @{0}.deleter
  def {0}(self):
    del self.operation.attributes["{1}"]
)Py";

constexpr const char *regionAccessorTemplate = R"PY(
  @builtins.property
  def {0}(self):
    return self.regions[{1}]
)PY";

static llvm::cl::OptionCategory
        clOpPythonBindingCat("Options for -gen-python-op-bindings");

static llvm::cl::opt<std::string>
        clDialectName("bind-dialect",
                      llvm::cl::desc("The dialect to run the generator for"),
                      llvm::cl::init(""), llvm::cl::cat(clOpPythonBindingCat));

static llvm::cl::opt<std::string> clDialectExtensionName(
        "dialect-extension", llvm::cl::desc("The prefix of the dialect extension"),
        llvm::cl::init(""), llvm::cl::cat(clOpPythonBindingCat));

using AttributeClasses = DenseMap<StringRef, StringRef>;

/// Checks whether `str` is a Python keyword or would shadow builtin function.
static bool isPythonReserved(StringRef str) {
    static llvm::StringSet<> reserved(
            {"and", "as", "assert", "break", "callable", "class",
             "continue", "def", "del", "elif", "else", "except",
             "finally", "for", "from", "global", "if", "import",
             "in", "is", "lambda", "nonlocal", "not", "or",
             "pass", "raise", "return", "issubclass", "try", "type",
             "while", "with", "yield"});
    return reserved.contains(str);
}

static bool isODSReserved(StringRef str) {
    static llvm::StringSet<> reserved(
            {"attributes", "create", "context", "ip", "operands", "print", "get_asm",
             "loc", "verify", "regions", "results", "self", "operation",
             "DIALECT_NAMESPACE", "OPERATION_NAME"});
    return str.startswith("_ods_") || str.endswith("_ods") ||
           reserved.contains(str);
}

static std::string sanitizeName(StringRef name) {
    if (isPythonReserved(name) || isODSReserved(name))
        return (name + "_").str();
    return name.str();
}

static std::string attrSizedTraitForKind(const char *kind) {
    return llvm::formatv("::mlir::OpTrait::AttrSized{0}{1}Segments",
                         llvm::StringRef(kind).take_front().upper(),
                         llvm::StringRef(kind).drop_front());
}

static void emitElementAccessors(
        const Operator &op, raw_ostream &os, const char *kind,
        llvm::function_ref<unsigned(const Operator &)> getNumVariableLength,
        llvm::function_ref<int(const Operator &)> getNumElements,
        llvm::function_ref<const NamedTypeConstraint &(const Operator &, int)>
        getElement) {
    assert(llvm::is_contained(
            llvm::SmallVector<StringRef, 2>{"operand", "result"}, kind) &&
           "unsupported kind");

    // Traits indicating how to process variadic elements.
    std::string sameSizeTrait =
            llvm::formatv("::mlir::OpTrait::SameVariadic{0}{1}Size",
                          llvm::StringRef(kind).take_front().upper(),
                          llvm::StringRef(kind).drop_front());
    std::string attrSizedTrait = attrSizedTraitForKind(kind);

    unsigned numVariableLength = getNumVariableLength(op);

    // If there is only one variable-length element group, its size can be
    // inferred from the total number of elements. If there are none, the
    // generation is straightforward.
    if (numVariableLength <= 1) {
        bool seenVariableLength = false;
        for (int i = 0, e = getNumElements(op); i < e; ++i) {
            const NamedTypeConstraint &element = getElement(op, i);
            if (element.isVariableLength())
                seenVariableLength = true;
            if (element.name.empty())
                continue;
            if (element.isVariableLength()) {
                os << llvm::formatv(element.isOptional() ? opOneOptionalTemplate
                                                         : opOneVariadicTemplate,
                                    sanitizeName(element.name), kind,
                                    getNumElements(op), i);
            } else if (seenVariableLength) {
                os << llvm::formatv(opSingleAfterVariableTemplate,
                                    sanitizeName(element.name), kind,
                                    getNumElements(op), i);
            } else {
                os << llvm::formatv(opSingleTemplate, sanitizeName(element.name), kind,
                                    i);
            }
        }
        return;
    }

    // Handle the operations where variadic groups have the same size.
    if (op.getTrait(sameSizeTrait)) {
        int numPrecedingSimple = 0;
        int numPrecedingVariadic = 0;
        for (int i = 0, e = getNumElements(op); i < e; ++i) {
            const NamedTypeConstraint &element = getElement(op, i);
            if (!element.name.empty()) {
                os << llvm::formatv(opVariadicEqualPrefixTemplate,
                                    sanitizeName(element.name), kind, numVariableLength,
                                    numPrecedingSimple, numPrecedingVariadic);
                os << llvm::formatv(element.isVariableLength()
                                    ? opVariadicEqualVariadicTemplate
                                    : opVariadicEqualSimpleTemplate,
                                    kind);
            }
            if (element.isVariableLength())
                ++numPrecedingVariadic;
            else
                ++numPrecedingSimple;
        }
        return;
    }

    // Handle the operations where the size of groups (variadic or not) is
    // provided as an attribute. For non-variadic elements, make sure to return
    // an element rather than a singleton container.
    if (op.getTrait(attrSizedTrait)) {
        for (int i = 0, e = getNumElements(op); i < e; ++i) {
            const NamedTypeConstraint &element = getElement(op, i);
            if (element.name.empty())
                continue;
            std::string trailing;
            if (!element.isVariableLength())
                trailing = "[0]";
            else if (element.isOptional())
                trailing = std::string(
                        llvm::formatv(opVariadicSegmentOptionalTrailingTemplate, kind));
            os << llvm::formatv(opVariadicSegmentTemplate, sanitizeName(element.name),
                                kind, i, trailing);
        }
        return;
    }

    llvm::PrintFatalError("unsupported " + llvm::Twine(kind) + " structure");
}

static int getNumOperands(const Operator &op) { return op.getNumOperands(); }

static const NamedTypeConstraint &getOperand(const Operator &op, int i) {
    return op.getOperand(i);
}

static int getNumResults(const Operator &op) { return op.getNumResults(); }

static const NamedTypeConstraint &getResult(const Operator &op, int i) {
    return op.getResult(i);
}

static void emitOperandAccessors(const Operator &op, raw_ostream &os) {
    auto getNumVariableLengthOperands = [](const Operator &oper) {
        return oper.getNumVariableLengthOperands();
    };
    emitElementAccessors(op, os, "operand", getNumVariableLengthOperands,
                         getNumOperands, getOperand);
}

static void emitResultAccessors(const Operator &op, raw_ostream &os) {
    auto getNumVariableLengthResults = [](const Operator &oper) {
        return oper.getNumVariableLengthResults();
    };
    emitElementAccessors(op, os, "result", getNumVariableLengthResults,
                         getNumResults, getResult);
}

static void emitAttributeAccessors(const Operator &op, raw_ostream &os) {
    for (const auto &namedAttr: op.getAttributes()) {
        // Skip "derived" attributes because they are just C++ functions that we
        // don't currently expose.
        if (namedAttr.attr.isDerivedAttr())
            continue;

        if (namedAttr.name.empty())
            continue;

        std::string sanitizedName = sanitizeName(namedAttr.name);

        // Unit attributes are handled specially.
        if (namedAttr.attr.getStorageType().trim().equals("::mlir::UnitAttr")) {
            os << llvm::formatv(unitAttributeGetterTemplate, sanitizedName,
                                namedAttr.name);
            os << llvm::formatv(unitAttributeSetterTemplate, sanitizedName,
                                namedAttr.name);
            os << llvm::formatv(attributeDeleterTemplate, sanitizedName,
                                namedAttr.name);
            continue;
        }

        if (namedAttr.attr.isOptional()) {
            os << llvm::formatv(optionalAttributeGetterTemplate, sanitizedName,
                                namedAttr.name);
            os << llvm::formatv(optionalAttributeSetterTemplate, sanitizedName,
                                namedAttr.name);
            os << llvm::formatv(attributeDeleterTemplate, sanitizedName,
                                namedAttr.name);
        } else {
            os << llvm::formatv(attributeGetterTemplate, sanitizedName,
                                namedAttr.name);
            os << llvm::formatv(attributeSetterTemplate, sanitizedName,
                                namedAttr.name);
            // Non-optional attributes cannot be deleted.
        }
    }
}

constexpr const char *initTemplate = R"Py(
  def __init__(self, {0}):
    operands = []
    results = []
    attributes = {{}
    regions = None
    {1}
    super().__init__(self.build_generic({2}))
)Py";

constexpr const char *singleOperandAppendTemplate =
        "operands.append(_get_op_result_or_value({0}))";
constexpr const char *singleResultAppendTemplate = "results.append({0})";

constexpr const char *optionalAppendOperandTemplate =
        "if {0} is not None: operands.append(_get_op_result_or_value({0}))";
constexpr const char *optionalAppendAttrSizedOperandsTemplate =
        "operands.append(_get_op_result_or_value({0}) if {0} is not None else "
        "None)";
constexpr const char *optionalAppendResultTemplate =
        "if {0} is not None: results.append({0})";

constexpr const char *multiOperandAppendTemplate =
        "operands.extend(_get_op_results_or_values({0}))";
constexpr const char *multiOperandAppendPackTemplate =
        "operands.append(_get_op_results_or_values({0}))";
constexpr const char *multiResultAppendTemplate = "results.extend({0})";

constexpr const char *initAttributeWithBuilderTemplate =
        R"Py(attributes["{1}"] = ({0} if (
    issubclass(type({0}), _ods_ir.Attribute) or
    not _ods_ir.AttrBuilder.contains('{2}')) else
      _ods_ir.AttrBuilder.get('{2}')({0}, context=_ods_context)))Py";

constexpr const char *initOptionalAttributeWithBuilderTemplate =
        R"Py(if {0} is not None: attributes["{1}"] = ({0} if (
        issubclass(type({0}), _ods_ir.Attribute) or
        not _ods_ir.AttrBuilder.contains('{2}')) else
          _ods_ir.AttrBuilder.get('{2}')({0}, context=_ods_context)))Py";

constexpr const char *initUnitAttributeTemplate =
        R"Py(if bool({1}): attributes["{0}"] = _ods_ir.UnitAttr.get(
      _ods_get_default_loc_context(loc)))Py";

constexpr const char *initSuccessorsTemplate = R"Py(_ods_successors = {0})Py";

constexpr const char *addSuccessorTemplate = R"Py(_ods_successors.{0}({1}))Py";

static bool hasSameArgumentAndResultTypes(const Operator &op) {
    return op.getTrait("::mlir::OpTrait::SameOperandsAndResultType") &&
           op.getNumVariableLengthResults() == 0;
}

static bool hasFirstAttrDerivedResultTypes(const Operator &op) {
    return op.getTrait("::mlir::OpTrait::FirstAttrDerivedResultType") &&
           op.getNumVariableLengthResults() == 0;
}

static bool hasInferTypeInterface(const Operator &op) {
    return op.getTrait("::mlir::InferTypeOpInterface::Trait") &&
           op.getNumRegions() == 0;
}

static bool canInferType(const Operator &op) {
    return hasSameArgumentAndResultTypes(op) ||
           hasFirstAttrDerivedResultTypes(op) || hasInferTypeInterface(op);
}

static void
populateBuilderArgsResults(const Operator &op,
                           llvm::SmallVectorImpl<std::string> &builderArgs) {
    if (canInferType(op))
        return;

    for (int i = 0, e = op.getNumResults(); i < e; ++i) {
        std::string name = op.getResultName(i).str();
        if (name.empty()) {
            if (op.getNumResults() == 1) {
                // Special case for one result, make the default name be 'result'
                // to properly match the built-in result accessor.
                name = "result";
            } else {
                name = llvm::formatv("_gen_res_{0}", i);
            }
        }
        name = sanitizeName(name);
        builderArgs.push_back(name);
    }
}

static void
populateBuilderArgs(const Operator &op,
                    llvm::SmallVectorImpl<std::string> &builderArgs,
                    llvm::SmallVectorImpl<std::string> &operandNames,
                    llvm::SmallVectorImpl<std::string> &successorArgNames) {

    for (int i = 0, e = op.getNumArgs(); i < e; ++i) {
        std::string name = op.getArgName(i).str();
        if (name.empty())
            name = llvm::formatv("_gen_arg_{0}", i);
        name = sanitizeName(name);
        builderArgs.push_back(name);
        if (!op.getArg(i).is<NamedAttribute *>())
            operandNames.push_back(name);
    }
}

static void populateBuilderArgsSuccessors(
        const Operator &op, llvm::SmallVectorImpl<std::string> &builderArgs,
        llvm::SmallVectorImpl<std::string> &successorArgNames) {

    for (int i = 0, e = op.getNumSuccessors(); i < e; ++i) {
        NamedSuccessor successor = op.getSuccessor(i);
        std::string name = std::string(successor.name);
        if (name.empty())
            name = llvm::formatv("_gen_successor_{0}", i);
        name = sanitizeName(name);
        builderArgs.push_back(name);
        successorArgNames.push_back(name);
    }
}

static void
populateBuilderLinesAttr(const Operator &op,
                         llvm::ArrayRef<std::string> argNames,
                         llvm::SmallVectorImpl<std::string> &builderLines) {
    builderLines.push_back("_ods_context = _ods_get_default_loc_context(loc)");
    for (int i = 0, e = op.getNumArgs(); i < e; ++i) {
        Argument arg = op.getArg(i);
        auto *attribute = llvm::dyn_cast_if_present<NamedAttribute *>(arg);
        if (!attribute)
            continue;

        // Unit attributes are handled specially.
        if (attribute->attr.getStorageType().trim().equals("::mlir::UnitAttr")) {
            builderLines.push_back(llvm::formatv(initUnitAttributeTemplate,
                                                 attribute->name, argNames[i]));
            continue;
        }

        builderLines.push_back(llvm::formatv(
                attribute->attr.isOptional() || attribute->attr.hasDefaultValue()
                ? initOptionalAttributeWithBuilderTemplate
                : initAttributeWithBuilderTemplate,
                argNames[i], attribute->name, attribute->attr.getAttrDefName()));
    }
}

static void populateBuilderLinesSuccessors(
        const Operator &op, llvm::ArrayRef<std::string> successorArgNames,
        llvm::SmallVectorImpl<std::string> &builderLines) {
    if (successorArgNames.empty()) {
        builderLines.push_back(llvm::formatv(initSuccessorsTemplate, "None"));
        return;
    }

    builderLines.push_back(llvm::formatv(initSuccessorsTemplate, "[]"));
    for (int i = 0, e = successorArgNames.size(); i < e; ++i) {
        auto &argName = successorArgNames[i];
        const NamedSuccessor &successor = op.getSuccessor(i);
        builderLines.push_back(
                llvm::formatv(addSuccessorTemplate,
                              successor.isVariadic() ? "extend" : "append", argName));
    }
}

static void
populateBuilderLinesOperand(const Operator &op,
                            llvm::ArrayRef<std::string> names,
                            llvm::SmallVectorImpl<std::string> &builderLines) {
    bool sizedSegments = op.getTrait(attrSizedTraitForKind("operand")) != nullptr;

    // For each element, find or generate a name.
    for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
        const NamedTypeConstraint &element = op.getOperand(i);
        std::string name = names[i];

        // Choose the formatting string based on the element kind.
        llvm::StringRef formatString;
        if (!element.isVariableLength()) {
            formatString = singleOperandAppendTemplate;
        } else if (element.isOptional()) {
            if (sizedSegments) {
                formatString = optionalAppendAttrSizedOperandsTemplate;
            } else {
                formatString = optionalAppendOperandTemplate;
            }
        } else {
            assert(element.isVariadic() && "unhandled element group type");
            // If emitting with sizedSegments, then we add the actual list-typed
            // element. Otherwise, we extend the actual operands.
            if (sizedSegments) {
                formatString = multiOperandAppendPackTemplate;
            } else {
                formatString = multiOperandAppendTemplate;
            }
        }

        builderLines.push_back(llvm::formatv(formatString.data(), name));
    }
}

constexpr const char *deriveTypeFromAttrTemplate =
        R"PY(_ods_result_type_source_attr = attributes["{0}"]
_ods_derived_result_type = (
    _ods_ir.TypeAttr(_ods_result_type_source_attr).value
    if _ods_ir.TypeAttr.isinstance(_ods_result_type_source_attr) else
    _ods_result_type_source_attr.type))PY";

constexpr const char *appendSameResultsTemplate = "results.extend([{0}] * {1})";

static void appendLineByLine(StringRef string,
                             llvm::SmallVectorImpl<std::string> &builderLines) {

    std::pair<StringRef, StringRef> split = std::make_pair(string, string);
    do {
        split = split.second.split('\n');
        builderLines.push_back(split.first.str());
    } while (!split.second.empty());
}

static void
populateBuilderLinesResult(const Operator &op,
                           llvm::ArrayRef<std::string> names,
                           llvm::SmallVectorImpl<std::string> &builderLines) {
    bool sizedSegments = op.getTrait(attrSizedTraitForKind("result")) != nullptr;

    if (hasSameArgumentAndResultTypes(op)) {
        builderLines.push_back(llvm::formatv(
                appendSameResultsTemplate, "operands[0].type", op.getNumResults()));
        return;
    }

    if (hasFirstAttrDerivedResultTypes(op)) {
        const NamedAttribute &firstAttr = op.getAttribute(0);
        assert(!firstAttr.name.empty() && "unexpected empty name for the attribute "
                                          "from which the type is derived");
        appendLineByLine(
                llvm::formatv(deriveTypeFromAttrTemplate, firstAttr.name).str(),
                builderLines);
        builderLines.push_back(llvm::formatv(appendSameResultsTemplate,
                                             "_ods_derived_result_type",
                                             op.getNumResults()));
        return;
    }

    if (hasInferTypeInterface(op))
        return;

    // For each element, find or generate a name.
    for (int i = 0, e = op.getNumResults(); i < e; ++i) {
        const NamedTypeConstraint &element = op.getResult(i);
        std::string name = names[i];

        // Choose the formatting string based on the element kind.
        llvm::StringRef formatString;
        if (!element.isVariableLength()) {
            formatString = singleResultAppendTemplate;
        } else if (element.isOptional()) {
            formatString = optionalAppendResultTemplate;
        } else {
            assert(element.isVariadic() && "unhandled element group type");
            // If emitting with sizedSegments, then we add the actual list-typed
            // element. Otherwise, we extend the actual operands.
            if (sizedSegments) {
                formatString = singleResultAppendTemplate;
            } else {
                formatString = multiResultAppendTemplate;
            }
        }

        builderLines.push_back(llvm::formatv(formatString.data(), name));
    }
}

static void
populateBuilderRegions(const Operator &op,
                       llvm::SmallVectorImpl<std::string> &builderArgs,
                       llvm::SmallVectorImpl<std::string> &builderLines) {
    if (op.hasNoVariadicRegions())
        return;

    // This is currently enforced when Operator is constructed.
    assert(op.getNumVariadicRegions() == 1 &&
           op.getRegion(op.getNumRegions() - 1).isVariadic() &&
           "expected the last region to be varidic");

    const NamedRegion &region = op.getRegion(op.getNumRegions() - 1);
    std::string name =
            ("num_" + region.name.take_front().lower() + region.name.drop_front())
                    .str();
    builderArgs.push_back(name);
    builderLines.push_back(
            llvm::formatv("regions = {0} + {1}", op.getNumRegions() - 1, name));
}

static void emitDefaultOpBuilder(const Operator &op, raw_ostream &os) {
    // If we are asked to skip default builders, comply.
    if (op.skipDefaultBuilders())
        return;

    llvm::SmallVector<std::string> builderArgs;
    llvm::SmallVector<std::string> builderLines;
    llvm::SmallVector<std::string> operandArgNames;
    llvm::SmallVector<std::string> successorArgNames;
    builderArgs.reserve(op.getNumOperands() + op.getNumResults() +
                        op.getNumNativeAttributes() + op.getNumSuccessors());
    populateBuilderArgsResults(op, builderArgs);
    size_t numResultArgs = builderArgs.size();
    populateBuilderArgs(op, builderArgs, operandArgNames, successorArgNames);
    size_t numOperandAttrArgs = builderArgs.size() - numResultArgs;
    populateBuilderArgsSuccessors(op, builderArgs, successorArgNames);

    populateBuilderLinesOperand(op, operandArgNames, builderLines);
    populateBuilderLinesAttr(
            op, llvm::ArrayRef(builderArgs).drop_front(numResultArgs), builderLines);
    populateBuilderLinesResult(
            op, llvm::ArrayRef(builderArgs).take_front(numResultArgs), builderLines);
    populateBuilderLinesSuccessors(op, successorArgNames, builderLines);
    populateBuilderRegions(op, builderArgs, builderLines);

    // Layout of builderArgs vector elements:
    // [ result_args  operand_attr_args successor_args regions ]

    // Determine whether the argument corresponding to a given index into the
    // builderArgs vector is a python keyword argument or not.
    auto isKeywordArgFn = [&](size_t builderArgIndex) -> bool {
        // All result, successor, and region arguments are positional arguments.
        if ((builderArgIndex < numResultArgs) ||
            (builderArgIndex >= (numResultArgs + numOperandAttrArgs)))
            return false;
        // Keyword arguments:
        // - optional named attributes (including unit attributes)
        // - default-valued named attributes
        // - optional operands
        Argument a = op.getArg(builderArgIndex - numResultArgs);
        if (auto *nattr = llvm::dyn_cast_if_present<NamedAttribute *>(a))
            return (nattr->attr.isOptional() || nattr->attr.hasDefaultValue());
        if (auto *ntype = llvm::dyn_cast_if_present<NamedTypeConstraint *>(a))
            return ntype->isOptional();
        return false;
    };

    // StringRefs in functionArgs refer to strings allocated by builderArgs.
    llvm::SmallVector<llvm::StringRef> functionArgs;

    // Add positional arguments.
    for (size_t i = 0, cnt = builderArgs.size(); i < cnt; ++i) {
        if (!isKeywordArgFn(i))
            functionArgs.push_back(builderArgs[i]);
    }

    // Add a bare '*' to indicate that all following arguments must be keyword
    // arguments.
    functionArgs.push_back("*");

    // Add a default 'None' value to each keyword arg string, and then add to the
    // function args list.
    for (size_t i = 0, cnt = builderArgs.size(); i < cnt; ++i) {
        if (isKeywordArgFn(i)) {
            builderArgs[i].append("=None");
            functionArgs.push_back(builderArgs[i]);
        }
    }
    functionArgs.push_back("loc=None");
    functionArgs.push_back("ip=None");

    SmallVector<std::string> initArgs;
    initArgs.push_back("attributes=attributes");
    if (!hasInferTypeInterface(op))
        initArgs.push_back("results=results");
    initArgs.push_back("operands=operands");
    initArgs.push_back("successors=_ods_successors");
    initArgs.push_back("regions=regions");
    initArgs.push_back("loc=loc");
    initArgs.push_back("ip=ip");

    os << llvm::formatv(initTemplate, llvm::join(functionArgs, ", "),
                        llvm::join(builderLines, "\n    "),
                        llvm::join(initArgs, ", "));
}

static void emitSegmentSpec(
        const Operator &op, const char *kind,
        llvm::function_ref<int(const Operator &)> getNumElements,
        llvm::function_ref<const NamedTypeConstraint &(const Operator &, int)>
        getElement,
        raw_ostream &os) {
    std::string segmentSpec("[");
    for (int i = 0, e = getNumElements(op); i < e; ++i) {
        const NamedTypeConstraint &element = getElement(op, i);
        if (element.isOptional()) {
            segmentSpec.append("0,");
        } else if (element.isVariadic()) {
            segmentSpec.append("-1,");
        } else {
            segmentSpec.append("1,");
        }
    }
    segmentSpec.append("]");

    os << llvm::formatv(opClassSizedSegmentsTemplate, kind, segmentSpec);
}

static void emitRegionAttributes(const Operator &op, raw_ostream &os) {
    // Emit _ODS_REGIONS = (min_region_count, has_no_variadic_regions).
    // Note that the base OpView class defines this as (0, True).
    unsigned minRegionCount = op.getNumRegions() - op.getNumVariadicRegions();
    os << llvm::formatv(opClassRegionSpecTemplate, minRegionCount,
                        op.hasNoVariadicRegions() ? "True" : "False");
}

/// Emits named accessors to regions.
static void emitRegionAccessors(const Operator &op, raw_ostream &os) {
    for (const auto &en: llvm::enumerate(op.getRegions())) {
        const NamedRegion &region = en.value();
        if (region.name.empty())
            continue;

        assert((!region.isVariadic() || en.index() == op.getNumRegions() - 1) &&
               "expected only the last region to be variadic");
        os << llvm::formatv(regionAccessorTemplate, sanitizeName(region.name),
                            std::to_string(en.index()) +
                            (region.isVariadic() ? ":" : ""));
    }
}

static void emitOpBindings(const Operator &op, raw_ostream &os) {
    os << llvm::formatv(opClassTemplate, op.getCppClassName(),
                        op.getOperationName());

    // Sized segments.
    if (op.getTrait(attrSizedTraitForKind("operand")) != nullptr) {
        emitSegmentSpec(op, "OPERAND", getNumOperands, getOperand, os);
    }
    if (op.getTrait(attrSizedTraitForKind("result")) != nullptr) {
        emitSegmentSpec(op, "RESULT", getNumResults, getResult, os);
    }

    emitRegionAttributes(op, os);
    emitDefaultOpBuilder(op, os);
    emitOperandAccessors(op, os);
    emitAttributeAccessors(op, os);
    emitResultAccessors(op, os);
    emitRegionAccessors(op, os);
}

static bool emitAllOps(const llvm::RecordKeeper &records, raw_ostream &os) {
    if (clDialectName.empty())
        llvm::PrintFatalError("dialect name not provided");

    bool isExtension = !clDialectExtensionName.empty();
    os << llvm::formatv(fileHeader, isExtension
                                    ? clDialectExtensionName.getValue()
                                    : clDialectName.getValue());
    if (isExtension)
        os << llvm::formatv(dialectExtensionTemplate, clDialectName.getValue());
    else
        os << llvm::formatv(dialectClassTemplate, clDialectName.getValue());

//    for (const llvm::Record *rec: records.getAllDerivedDefinitions("Op")) {
//        Operator op(rec);
//        if (op.getDialectName() == clDialectName.getValue())
//            emitOpBindings(op, os);
//    }
    return false;
}

static GenRegistration
        genPythonBindings("gen-python-op-bindings",
                          "Generate Python bindings for MLIR Ops", &emitAllOps);
