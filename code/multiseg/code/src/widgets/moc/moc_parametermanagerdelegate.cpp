/****************************************************************************
** Meta object code from reading C++ file 'ParameterManagerDelegate.hpp'
**
** Created: Sun Sep 8 18:55:06 2013
**      by: The Qt Meta Object Compiler version 61 (Qt 4.5.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../widgets/ParameterManagerDelegate.hpp"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ParameterManagerDelegate.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 61
#error "This file was generated using the moc from 4.5.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_ParameterManagerDelegate[] = {

 // content:
       2,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors

       0        // eod
};

static const char qt_meta_stringdata_ParameterManagerDelegate[] = {
    "ParameterManagerDelegate\0"
};

const QMetaObject ParameterManagerDelegate::staticMetaObject = {
    { &QItemDelegate::staticMetaObject, qt_meta_stringdata_ParameterManagerDelegate,
      qt_meta_data_ParameterManagerDelegate, 0 }
};

const QMetaObject *ParameterManagerDelegate::metaObject() const
{
    return &staticMetaObject;
}

void *ParameterManagerDelegate::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_ParameterManagerDelegate))
        return static_cast<void*>(const_cast< ParameterManagerDelegate*>(this));
    return QItemDelegate::qt_metacast(_clname);
}

int ParameterManagerDelegate::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QItemDelegate::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    return _id;
}
static const uint qt_meta_data_Slider[] = {

 // content:
       2,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   12, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors

 // slots: signature, parameters, type, tag, flags
      14,    8,    7,    7, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_Slider[] = {
    "Slider\0\0value\0updateModel(int)\0"
};

const QMetaObject Slider::staticMetaObject = {
    { &QSlider::staticMetaObject, qt_meta_stringdata_Slider,
      qt_meta_data_Slider, 0 }
};

const QMetaObject *Slider::metaObject() const
{
    return &staticMetaObject;
}

void *Slider::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_Slider))
        return static_cast<void*>(const_cast< Slider*>(this));
    return QSlider::qt_metacast(_clname);
}

int Slider::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QSlider::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: updateModel((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 1;
    }
    return _id;
}
static const uint qt_meta_data_SpinBox[] = {

 // content:
       2,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   12, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors

 // slots: signature, parameters, type, tag, flags
      15,    9,    8,    8, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_SpinBox[] = {
    "SpinBox\0\0value\0updateModel(double)\0"
};

const QMetaObject SpinBox::staticMetaObject = {
    { &QDoubleSpinBox::staticMetaObject, qt_meta_stringdata_SpinBox,
      qt_meta_data_SpinBox, 0 }
};

const QMetaObject *SpinBox::metaObject() const
{
    return &staticMetaObject;
}

void *SpinBox::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_SpinBox))
        return static_cast<void*>(const_cast< SpinBox*>(this));
    return QDoubleSpinBox::qt_metacast(_clname);
}

int SpinBox::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDoubleSpinBox::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: updateModel((*reinterpret_cast< double(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 1;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
