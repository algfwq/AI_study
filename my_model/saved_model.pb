�u
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*
2.15.0-rc12v2.15.0-rc0-8-g2a4ec940bac8�]

VariableVarHandleOp*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
�

Variable_1VarHandleOp*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape: *
shared_name
Variable_1
a
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
: *
dtype0
�

Variable_2VarHandleOp*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape: *
shared_name
Variable_2
a
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
: *
dtype0
Z
serving_default_xPlaceholder*
_output_shapes
:*
dtype0*
shape:
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_x
Variable_2
Variable_1Variable*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_13800137

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
7
a
b
c
__call__

signatures*
@:
VARIABLE_VALUE
Variable_2a/.ATTRIBUTES/VARIABLE_VALUE*
@:
VARIABLE_VALUE
Variable_1b/.ATTRIBUTES/VARIABLE_VALUE*
>8
VARIABLE_VALUEVariablec/.ATTRIBUTES/VARIABLE_VALUE*

trace_0* 

serving_default* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename
Variable_2
Variable_1VariableConst*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_13800177
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
Variable_2
Variable_1Variable*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_13800195�K
�%
�
!__inference__traced_save_13800177
file_prefix+
!read_disablecopyonread_variable_2: -
#read_1_disablecopyonread_variable_1: +
!read_2_disablecopyonread_variable: 
savev2_const

identity_7��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: s
Read/DisableCopyOnReadDisableCopyOnRead!read_disablecopyonread_variable_2"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp!read_disablecopyonread_variable_2^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0a
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
: w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_variable_1"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_variable_1^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: u
Read_2/DisableCopyOnReadDisableCopyOnRead!read_2_disablecopyonread_variable"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp!read_2_disablecopyonread_variable^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�Ba/.ATTRIBUTES/VARIABLE_VALUEBb/.ATTRIBUTES/VARIABLE_VALUEBc/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHu
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h

Identity_6Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: S

Identity_7IdentityIdentity_6:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_7Identity_7:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
&__inference_signature_wrapper_13800137
x
unknown: 
	unknown_0: 
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference___call___13800125b
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:: : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
13800133:($
"
_user_specified_name
13800131:($
"
_user_specified_name
13800129:= 9

_output_shapes
:

_user_specified_namex
�
�
$__inference__traced_restore_13800195
file_prefix%
assignvariableop_variable_2: '
assignvariableop_1_variable_1: %
assignvariableop_2_variable: 

identity_4��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�Ba/.ATTRIBUTES/VARIABLE_VALUEBb/.ATTRIBUTES/VARIABLE_VALUEBc/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHx
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_2Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_1Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variableIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_3Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_4IdentityIdentity_3:output:0^NoOp_1*
T0*
_output_shapes
: a
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2*
_output_shapes
 "!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22$
AssignVariableOpAssignVariableOp:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
__inference___call___13800125
x!
readvariableop_resource: #
readvariableop_1_resource: '
add_1_readvariableop_resource: 
identity��ReadVariableOp�ReadVariableOp_1�add_1/ReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0J
mulMulReadVariableOp:value:0x*
T0*
_output_shapes
:=
mul_1Mulmul:z:0x*
T0*
_output_shapes
:b
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0N
mul_2MulReadVariableOp_1:value:0x*
T0*
_output_shapes
:G
addAddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:j
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
: *
dtype0Z
add_1AddV2add:z:0add_1/ReadVariableOp:value:0*
T0*
_output_shapes
:K
IdentityIdentity	add_1:z:0^NoOp*
T0*
_output_shapes
:]
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:: : : 2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:= 9

_output_shapes
:

_user_specified_namex"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_defaultq
"
x
serving_default_x:0/
output_0#
StatefulPartitionedCall:0tensorflow/serving/predict:�
Q
a
b
c
__call__

signatures"
_generic_user_object
: 2Variable
: 2Variable
: 2Variable
�
trace_02�
__inference___call___13800125�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
�ztrace_0
,
serving_default"
signature_map
�B�
__inference___call___13800125x"�
���
FullArgSpec
args�
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_13800137x"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
jx
kwonlydefaults
 
annotations� *
 [
__inference___call___13800125:�
�
�
x
� "�
unknown{
&__inference_signature_wrapper_13800137Q"�
� 
�

x�
x"&�#
!
output_0�
output_0