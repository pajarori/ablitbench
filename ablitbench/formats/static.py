from __future__ import annotations

import ast
import importlib
from typing import Any

EXECUTORS = {
    "builtins.eval",
    "builtins.exec",
    "builtins.__import__",
    "eval",
    "exec",
    "os.system",
    "os.popen",
    "subprocess.run",
    "subprocess.call",
    "subprocess.Popen",
    "subprocess.check_call",
    "subprocess.check_output",
    "pickle.load",
    "pickle.loads",
    "dill.load",
    "dill.loads",
    "torch.load",
    "__import__",
    "importlib.import_module",
}


class PythonInspector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.bindings: dict[str, str | None] = {}
        self.observations: list[dict[str, Any]] = []

    def resolve(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return self.bindings.get(node.id, node.id if node.id in EXECUTORS else None)
        if isinstance(node, ast.Attribute):
            base = self.resolve(node.value)
            name = base + "." + node.attr if base else None
            return self.bindings.get(name, name) if name else None
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            base = self.resolve(node.value)
            if base and base.endswith(".__dict__") and isinstance(node.slice.value, str):
                return base.removesuffix(".__dict__") + "." + node.slice.value
        if (
            isinstance(node, ast.Call)
            and self.resolve(node.func) == "functools.partial"
            and node.args
        ):
            return self.resolve(node.args[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "getattr" and "getattr" not in self.bindings and len(node.args) == 2:
                base = self.resolve(node.args[0])
                attribute = node.args[1]
                if (
                    base
                    and isinstance(attribute, ast.Constant)
                    and isinstance(attribute.value, str)
                ):
                    return base + "." + attribute.value
        return None

    def bind(self, target: ast.expr, value: str | None = None) -> None:
        if isinstance(target, ast.Name):
            self.bindings[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self.bind(item)
        elif isinstance(target, ast.Attribute):
            name = self.resolve(target)
            if name:
                self.bindings[name] = value

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.bindings[alias.asname or alias.name.split(".")[0]] = (
                alias.name if alias.asname else alias.name.split(".")[0]
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                self.observations.append(
                    {"line": node.lineno, "gap": "Wildcard import bindings unresolved"}
                )
            else:
                self.bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def visit_Call(self, node: ast.Call) -> None:
        name = self.resolve(node.func)
        if name in EXECUTORS:
            self.observations.append(
                {
                    "line": node.lineno,
                    "call": name,
                    "issue": "execution-capable loader call",
                    "classification": "capability indicator",
                    "interpretation": "Static call; reachability and malicious intent unproven",
                }
            )
        dynamic_attribute = (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and node.args
            and self.resolve(node.args[0]) is not None
            and self.resolve(node) is None
        )
        if name is None and (isinstance(node.func, (ast.Call, ast.Subscript)) or dynamic_attribute):
            self.observations.append(
                {"line": node.lineno, "gap": "Dynamic callable resolution not statically validated"}
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        value = self.resolve(node.value)
        for target in node.targets:
            if (
                isinstance(target, (ast.Tuple, ast.List))
                and isinstance(node.value, (ast.Tuple, ast.List))
                and len(target.elts) == len(node.value.elts)
            ):
                for left, right in zip(target.elts, node.value.elts, strict=True):
                    self.bind(left, self.resolve(right))
            else:
                self.bind(target, value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value:
            self.visit(node.value)
            self.bind(node.target, self.resolve(node.value))

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self.bind(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self.bind(node.target, self.resolve(node.value))

    def scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        outer = self.bindings
        self.bindings = outer.copy()
        args = node.args
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            self.bindings[arg.arg] = None
        for variadic in (args.vararg, args.kwarg):
            if variadic:
                self.bindings[variadic.arg] = None
        body = node.body if isinstance(node.body, list) else [node.body]
        pending: list[ast.AST] = list(body)
        while pending:
            candidate = pending.pop()
            if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.bindings[candidate.name] = None
                continue
            if isinstance(candidate, ast.Lambda):
                continue
            if isinstance(candidate, ast.Name) and isinstance(candidate.ctx, (ast.Store, ast.Del)):
                self.bindings[candidate.id] = None
            if isinstance(candidate, (ast.Import, ast.ImportFrom)):
                for alias in candidate.names:
                    self.bindings[alias.asname or alias.name.split(".")[0]] = None
            pending.extend(ast.iter_child_nodes(candidate))
        for statement in body:
            self.visit(statement)
        self.bindings = outer

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for expression in (
            node.decorator_list
            + node.args.defaults
            + [value for value in node.args.kw_defaults if value is not None]
        ):
            self.visit(expression)
        self.bindings[node.name] = None
        self.scoped(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        for expression in (
            node.decorator_list
            + node.args.defaults
            + [value for value in node.args.kw_defaults if value is not None]
        ):
            self.visit(expression)
        self.bindings[node.name] = None
        self.scoped(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.scoped(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bindings[node.name] = None
        outer = self.bindings
        self.bindings = outer.copy()
        self.generic_visit(node)
        self.bindings = outer

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        before = self.bindings.copy()
        for statement in node.body:
            self.visit(statement)
        after = self.bindings.copy()
        self.bindings = before.copy()
        for statement in node.orelse:
            self.visit(statement)
        other = self.bindings
        self.bindings = {
            key: after.get(key) if after.get(key) == other.get(key) else None
            for key in after.keys() | other.keys()
        }
        for key in self.bindings:
            if after.get(key) != other.get(key) and (after.get(key) or other.get(key)):
                self.observations.append(
                    {"line": node.lineno, "gap": f"Conditional binding unresolved: {key}"}
                )

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self.bind(node.target)
        for statement in node.body + node.orelse:
            self.visit(statement)

    def comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        outer = self.bindings
        self.bindings = outer.copy()
        for generator in node.generators:
            self.visit(generator.iter)
            self.bind(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.bindings = outer

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.comprehension(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self.bind(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.bindings[node.name] = None
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            self.bind(target)

    def visit_Global(self, node: ast.Global) -> None:
        self.observations.append(
            {"line": node.lineno, "gap": "Global rebinding requires runtime scope analysis"}
        )

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.observations.append(
            {"line": node.lineno, "gap": "Nonlocal rebinding requires runtime scope analysis"}
        )


def analyze_python(text: str) -> list[dict[str, Any]]:
    inspector = PythonInspector()
    inspector.visit(ast.parse(text))
    return inspector.observations


def analyze_template(text: str) -> list[dict[str, Any]]:
    try:
        jinja = importlib.import_module("jinja2")
        nodes = importlib.import_module("jinja2.nodes")
    except ModuleNotFoundError:
        return [{"gap": "Chat template AST requires Jinja2 parser"}]
    try:
        tree = jinja.Environment().parse(text)
    except jinja.TemplateSyntaxError as error:
        return [{"gap": "Malformed chat template", "line": error.lineno}]
    observations: list[dict[str, Any]] = []
    known_filters = jinja.Environment().filters
    for node in tree.find_all(nodes.Filter):
        if node.name == "attr" and node.args:
            attribute = getattr(node.args[0], "value", None)
            if isinstance(attribute, str) and attribute.startswith("_"):
                observations.append(
                    {
                        "issue": "private attribute access in chat template",
                        "line": node.lineno,
                        "attribute": attribute,
                    }
                )
        if node.name not in known_filters:
            observations.append(
                {
                    "issue": "unresolved filter in chat template",
                    "line": node.lineno,
                    "filter": node.name,
                    "coverage_gap": "Custom template filter semantics unavailable",
                }
            )
    for node in tree.find_all((nodes.Getattr, nodes.Getitem)):
        attribute = (
            node.attr if isinstance(node, nodes.Getattr) else getattr(node.arg, "value", None)
        )
        if isinstance(attribute, str) and attribute.startswith("_"):
            observations.append(
                {
                    "issue": "private attribute access in chat template",
                    "line": node.lineno,
                    "attribute": attribute,
                }
            )
    for node in tree.find_all(nodes.Call):
        target = node.node
        name = target.name if isinstance(target, nodes.Name) else getattr(target, "attr", None)
        if name not in {
            "range",
            "dict",
            "namespace",
            "raise_exception",
            "strftime_now",
            "items",
            "keys",
            "values",
            "get",
            "strip",
            "lstrip",
            "rstrip",
            "replace",
            "lower",
            "upper",
            "startswith",
            "endswith",
        }:
            observations.append(
                {
                    "issue": "unresolved callable in chat template",
                    "line": node.lineno,
                    "call": name,
                    "coverage_gap": "Template callable semantics require runtime context",
                }
            )
    for node in tree.find_all((nodes.Include, nodes.Import, nodes.FromImport, nodes.Extends)):
        observations.append(
            {"gap": "External template dependency not resolved", "line": node.lineno}
        )
    return observations
