# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

import json
import sys
from typing import Any, List, Dict, Union

from PySide6.QtWidgets import QTreeView, QApplication, QHeaderView
from PySide6.QtCore import QAbstractItemModel, QModelIndex, QObject, Qt, QFileInfo, Signal
from PySide6.QtWidgets import QComboBox, QItemDelegate, QStyledItemDelegate
from PySide6.QtGui import QStandardItemModel

class TreeItem:
    """A Json item corresponding to a line in QTreeView"""

    def __init__(self, parent: "TreeItem" = None):
        self._parent = parent
        self._key = ""
        self._value = ""
        self._value_type = None
        self._children = []

    def appendChild(self, item: "TreeItem"):
        """Add item as a child"""
        self._children.append(item)

    def child(self, row: int) -> "TreeItem":
        """Return the child of the current item from the given row"""
        return self._children[row]

    def parent(self) -> "TreeItem":
        """Return the parent of the current item"""
        return self._parent

    def childCount(self) -> int:
        """Return the number of children of the current item"""
        return len(self._children)

    def row(self) -> int:
        """Return the row where the current item occupies in the parent"""
        return self._parent._children.index(self) if self._parent else 0

    @property
    def key(self) -> str:
        """Return the key name"""
        return self._key

    @key.setter
    def key(self, key: str):
        """Set key name of the current item"""
        self._key = key

    @property
    def value(self) -> str:
        """Return the value name of the current item"""
        return self._value

    @value.setter
    def value(self, value: str):
        """Set value name of the current item"""
        self._value = value

    @property
    def value_type(self):
        """Return the python type of the item's value."""
        return self._value_type

    @value_type.setter
    def value_type(self, value):
        """Set the python type of the item's value."""
        self._value_type = value

    @classmethod
    def load(
        cls, value: Union[List, Dict], parent: "TreeItem" = None, sort=False, excludedKeys:dict={},
    ) -> "TreeItem":
        """Create a 'root' TreeItem from a nested list or a nested dictonary

        Examples:
            with open("file.json") as file:
                data = json.dump(file)
                root = TreeItem.load(data)

        This method is a recursive function that calls itself.

        Returns:
            TreeItem: TreeItem
        """
        rootItem = TreeItem(parent)
        rootItem.key = "root"

        if isinstance(value, dict):
            items = sorted(value.items()) if sort else value.items()

            for key, value in items:
                allKeys = []
                for category,category_val in excludedKeys.items():
                    # print("category_val:",category_val)
                    if isinstance(category_val,dict):
                        for i in category_val.values():
                            if isinstance(i,list):
                                allKeys+=i
                            else:
                                allKeys.append(i)
                    else:
                        allKeys+=category_val
                # print("allKeys: ",allKeys)
                if key not in allKeys:
                    # print(key)
                    child = cls.load(value, rootItem,sort=sort,excludedKeys=excludedKeys)
                    child.key = key
                    child.value_type = type(value)
                    rootItem.appendChild(child)

        elif isinstance(value, list):
            for index, value in enumerate(value):
                child = cls.load(value, rootItem,sort=sort,excludedKeys=excludedKeys)
                child.key = index
                child.value_type = type(value)
                rootItem.appendChild(child)

        else:
            rootItem.value = value
            rootItem.value_type = type(value)

        return rootItem


class JsonModel(QAbstractItemModel):
    """ An editable model of Json data """
    def __init__(self, parent: QObject = None):
        super().__init__(parent)

        self._rootItem = TreeItem()
        self._headers = ("key", "type", "value")
        self.markedItem = []

    def clear(self):
        """ Clear data from the model """
        self.load({})

    def load(self, document: dict,excludedKeys:dict={}):
        """Load model from a nested dictionary returned by json.loads()

        Arguments:
            document (dict): JSON-compatible dictionary
        """

        assert isinstance(
            document, (dict, list, tuple)
        ), "`document` must be of dict, list or tuple, " f"not {type(document)}"

        self.beginResetModel()

        self._rootItem = TreeItem.load(document,excludedKeys=excludedKeys)
        self._rootItem.value_type = type(document)

        self.endResetModel()

        return True

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> Any:
        """Override from QAbstractItemModel

        Return data from a json item according index and role

        """
        if not index.isValid():
            return None

        item = index.internalPointer()

        if role == Qt.DisplayRole:
            if index.column() == 0:
                return item.key
            
            if index.column() == 1:
                return str(item.value_type.__name__)

            if index.column() == 2:
                return item.value

        elif role == Qt.EditRole:
            if index.column() == 2:
                return item.value

    def setData(self, index: QModelIndex, value: Any, role: Qt.ItemDataRole):
        """Override from QAbstractItemModel

        Set json item according index and role

        Args:
            index (QModelIndex)
            value (Any)
            role (Qt.ItemDataRole)

        """
        if role == Qt.EditRole:
            if index.column() == 2:
                item = index.internalPointer()
                item.value =eval(f"bool({value})") if item.value_type==bool else str(value)

                self.dataChanged.emit(index, index, [Qt.EditRole])

                return True

        return False

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole
    ):
        """Override from QAbstractItemModel

        For the JsonModel, it returns only data for columns (orientation = Horizontal)

        """
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return self._headers[section]

    def index(self, row: int, column: int, parent=QModelIndex()) -> QModelIndex:
        """Override from QAbstractItemModel

        Return index according row, column and parent

        """
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            parentItem = self._rootItem
        else:
            parentItem = parent.internalPointer()

        childItem = parentItem.child(row)
        if childItem:
            return self.createIndex(row, column, childItem)
        else:
            return QModelIndex()

    def parent(self, index: QModelIndex) -> QModelIndex:
        """Override from QAbstractItemModel

        Return parent index of index

        """

        if not index.isValid():
            return QModelIndex()

        childItem = index.internalPointer()
        parentItem = childItem.parent()

        if parentItem == self._rootItem:
            return QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)

    def rowCount(self, parent=QModelIndex()):
        """Override from QAbstractItemModel

        Return row count from parent index
        """
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parentItem = self._rootItem
        else:
            parentItem = parent.internalPointer()

        return parentItem.childCount()

    def columnCount(self, parent=QModelIndex()):
        """Override from QAbstractItemModel

        Return column number. For the model, it always return 2 columns
        """
        return 3

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """Override from QAbstractItemModel

        Return flags of index
        """
        flags = super(JsonModel, self).flags(index)

        if index.column() == 2:
            return Qt.ItemIsEditable | flags
        else:
            return flags

    def to_json(self, item=None):

        if item is None:
            item = self._rootItem

        nchild = item.childCount()

        if item.value_type is dict:
            document = {}
            for i in range(nchild):
                ch = item.child(i)
                document[ch.key] = self.to_json(ch)
            return document

        elif item.value_type == list:
            document = []
            for i in range(nchild):
                ch = item.child(i)
                document.append(self.to_json(ch))
            return document

        else:
            try:
                return item.value_type(item.value)
            except TypeError:
                return None

class JsonDelegate(QItemDelegate):

    def __init__(self,comboItem:dict,model:JsonModel=None,document:dict=None,metaKeys:dict={}):
        QItemDelegate.__init__(self)
        self.comboItem = comboItem
        self.model = model
        self.document=document.copy() # memory overwrite happened at here if not copy
        self.tItem = comboItem
        self.metaKeys = metaKeys
        # print("self.document in init: ",self.document)

    def createEditor(self, parent, option, index):
        # print(index.siblingAtColumn(0).data()=="device")
        if index.column() == 2:
            for section,val in self.comboItem.items():
                if isinstance(val,dict):
                    for k, v in val.items():
                        if index.siblingAtColumn(0).data()==k:
                            comboBox = QComboBox(parent)
                            # print("items:",self.comboItem[section][k])
                            for text in self.comboItem[section][k]:
                                # print(text)
                                comboBox.addItem(text, (index.row(), index.column()))
                            # print("section: ",section)
                            comboBox.currentTextChanged.connect(lambda val,section=section,key=k:(self.document[section].update(**{key:val}),self.metaKeys.update(**{key:[]}),self.metaKeys.update(**{key:[comboBox.itemText(i)if comboBox.itemText(i)!=comboBox.currentText()else"" for i in range(comboBox.count())]})))
                            comboBox.destroyed.connect(lambda key=k:(self.model.load(self.document,excludedKeys=self.metaKeys)))
                            return comboBox
        return super().createEditor(parent, option, index)

    def setModelData(self, editor, model, index):
        model.setData(index,editor.currentText() if isinstance(editor,QComboBox)else editor.text(), Qt.EditRole)



if __name__ == "__main__":

    app = QApplication(sys.argv)
    view = QTreeView()
    model = JsonModel()
    model.dataChanged.connect(lambda:json.dump(model.to_json(),open("changed.json","w"),indent=4))

    view.setModel(model)

    json_path = QFileInfo(__file__).absoluteDir().filePath("test.json")

    with open(json_path) as file:
        document = json.load(file)
        # print(document)
        excludedKeys = {
            "settings":{"device":["auto", "cuda", "cpu"],},
            "hyperparameters":{"optimizeralgorithm":["Adam","SGD","RMSprop"],},
        }
        view.setItemDelegateForColumn(2,JsonDelegate(excludedKeys,model,document))
        model.load(document,excludedKeys)
        view.expandAll()
        model.modelReset.connect(view.expandAll)

    view.show()
    view.header().setSectionResizeMode(0, QHeaderView.Stretch)
    view.setAlternatingRowColors(True)
    view.resize(500, 300)
    app.exec()