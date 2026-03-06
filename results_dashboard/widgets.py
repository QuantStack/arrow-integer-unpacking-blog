import ipywidgets as widgets
import traitlets


class MultiSelect(widgets.VBox):
    """Generic multi-select checkbox widget with .value and observe support."""

    value = traitlets.Tuple(help="Currently selected values")

    def __init__(self, options, default_filter=None, description="", **kwargs):
        """
        Parameters
        ----------
        options : iterable
            Available choices.
        default_filter : callable, optional
            fn(option) -> bool to set initial checked state. All checked if None.
        description : str, optional
            Label shown above the checkboxes.
        """
        self._checkboxes = {}
        for opt in options:
            checked = default_filter(opt) if default_filter else True
            cb = widgets.Checkbox(value=checked, description=str(opt), indent=False)
            cb._select_key = opt
            cb.observe(self._on_change, names="value")
            self._checkboxes[opt] = cb

        children = []
        if description:
            children.append(widgets.Label(description))
        children.append(
            widgets.HBox(
                list(self._checkboxes.values()),
                layout=widgets.Layout(flex_flow="row wrap"),
            )
        )

        super().__init__(children=children, **kwargs)
        self._sync_value()

    def _on_change(self, change):
        self._sync_value()

    def _sync_value(self):
        self.value = tuple(key for key, cb in self._checkboxes.items() if cb.value)


class GroupedMultiSelect(widgets.VBox):
    """Multi-select checkboxes grouped by category with .value and observe support.

    Parameters
    ----------
    groups : dict[str, list[str]]
        Mapping of category name to list of options.
    default_filter : callable, optional
        fn(category, option) -> bool for initial checked state. All checked if None.
    """

    value = traitlets.Dict(
        help="Currently selected values per group: {category: (selected, ...)}"
    )

    def __init__(self, groups, default_filter=None, **kwargs):
        self._groups = {}  # category -> {option: Checkbox}
        children = []

        for category, options in groups.items():
            checkboxes = {}
            for opt in options:
                checked = default_filter(category, opt) if default_filter else True
                cb = widgets.Checkbox(
                    value=checked,
                    description=str(opt),
                    indent=False,
                    layout=widgets.Layout(
                        width="auto", min_width="0", margin="0 0 0 12px"
                    ),
                )
                cb.observe(self._on_change, names="value")
                checkboxes[opt] = cb
            self._groups[category] = checkboxes

            children.append(widgets.Label(f"{category}:"))
            children.append(
                widgets.HBox(
                    list(checkboxes.values()),
                    layout=widgets.Layout(flex_flow="row wrap"),
                )
            )

        super().__init__(children=children, **kwargs)
        self._sync_value()

    def _on_change(self, change):
        self._sync_value()

    def _sync_value(self):
        self.value = {
            cat: tuple(opt for opt, cb in cbs.items() if cb.value)
            for cat, cbs in self._groups.items()
        }
