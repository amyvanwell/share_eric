��
��
.
Abs
x"T
y"T"
Ttype:

2	
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
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
�
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
,
Log
x"T
y"T"
Ttype:

2
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
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
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
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
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
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
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02unknown8��
G
ConstConst*
_output_shapes
: *
dtype0*
value	B :
p
Const_1Const*
_output_shapes
:*
dtype0*5
value,B*"                         
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  `B
�
Const_3Const*&
_output_shapes
:*
dtype0*A
value8B6"   �?  �?                        
�
Const_4Const*
_output_shapes

:8*
dtype0*�
value�B�8"�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
I
Const_5Const*
_output_shapes
: *
dtype0*
value	B :
�
@Adam/behavior_model_1/rank_similarity_1/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*Q
shared_nameB@Adam/behavior_model_1/rank_similarity_1/embedding_1/embeddings/v
�
TAdam/behavior_model_1/rank_similarity_1/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOp@Adam/behavior_model_1/rank_similarity_1/embedding_1/embeddings/v*
_output_shapes
:	�*
dtype0
�
@Adam/behavior_model_1/rank_similarity_1/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*Q
shared_nameB@Adam/behavior_model_1/rank_similarity_1/embedding_1/embeddings/m
�
TAdam/behavior_model_1/rank_similarity_1/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOp@Adam/behavior_model_1/rank_similarity_1/embedding_1/embeddings/m*
_output_shapes
:	�*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
�
exponential_similarity_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameexponential_similarity_1/beta
�
1exponential_similarity_1/beta/Read/ReadVariableOpReadVariableOpexponential_similarity_1/beta*
_output_shapes
: *
dtype0
�
exponential_similarity_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name exponential_similarity_1/gamma
�
2exponential_similarity_1/gamma/Read/ReadVariableOpReadVariableOpexponential_similarity_1/gamma*
_output_shapes
: *
dtype0
�
exponential_similarity_1/tauVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameexponential_similarity_1/tau
�
0exponential_similarity_1/tau/Read/ReadVariableOpReadVariableOpexponential_similarity_1/tau*
_output_shapes
: *
dtype0
�
Abehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:*R
shared_nameCAbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/w
�
Ubehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/w/Read/ReadVariableOpReadVariableOpAbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/w*
_output_shapes
:*
dtype0
r
minkowski_1/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameminkowski_1/rho
k
#minkowski_1/rho/Read/ReadVariableOpReadVariableOpminkowski_1/rho*
_output_shapes
: *
dtype0
�
9behavior_model_1/rank_similarity_1/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*J
shared_name;9behavior_model_1/rank_similarity_1/embedding_1/embeddings
�
Mbehavior_model_1/rank_similarity_1/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp9behavior_model_1/rank_similarity_1/embedding_1/embeddings*
_output_shapes
:	�*
dtype0
�
#serving_default_8rank2_stimulus_setPlaceholder*+
_output_shapes
:���������	*
dtype0	* 
shape:���������	
�
StatefulPartitionedCallStatefulPartitionedCall#serving_default_8rank2_stimulus_setConst_1Const9behavior_model_1/rank_similarity_1/embedding_1/embeddingsConst_5minkowski_1/rhoAbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/wexponential_similarity_1/betaexponential_similarity_1/tauexponential_similarity_1/gammaConst_4Const_3Const_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������8*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_219167

NoOpNoOp
�@
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*�?
value�?B�? B�?
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
behavior
		optimizer


signatures*
.
0
1
2
3
4
5*

0*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
trace_1
trace_2
trace_3* 
^
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11* 
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*percept

+kernel
,percept_adapter
-kernel_adapter
.
_z_q_shape
/
_z_r_shape*
X
0iter

1beta_1

2beta_2
	3decay
4learning_ratem�v�*

5serving_default* 
ys
VARIABLE_VALUE9behavior_model_1/rank_similarity_1/embedding_1/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEminkowski_1/rho&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/w&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEexponential_similarity_1/tau&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEexponential_similarity_1/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEexponential_similarity_1/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
'
0
1
2
3
4*

0*

60
71*
* 
* 
^
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11* 
^
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11* 
^
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11* 
^
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11* 
^
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11* 
^
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11* 
^
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11* 
^
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11* 
* 
* 
* 
* 
* 
* 
.
0
1
2
3
4
5*

0*
* 
�
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

=trace_0
>trace_1* 

?trace_0
@trace_1* 
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

embeddings*
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
Mdistance
N
similarity*
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
U	_all_keys
V_input_keys
Wgating_keys* 
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^	_all_keys
__input_keys
`gating_keys* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
^
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11* 
8
a	variables
b	keras_api
	ctotal
	dcount*
H
e	variables
f	keras_api
	gtotal
	hcount
i
_fn_kwargs*
'
0
1
2
3
4*
 
*0
+1
,2
-3*
* 
* 
* 
^
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11* 
^
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11* 
^
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11* 
^
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11* 

0*

0*
* 
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*
* 
* 
'
0
1
2
3
4*
* 
* 
�
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*
* 
* 
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses
rho
w*
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
tau
	gamma
beta*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

c0
d1*

a	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

g0
h1*

e	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
'
0
1
2
3
4*

M0
N1*
* 
* 
* 

0
1*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*
* 
* 

0
1
2*
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 

0
1
2*
* 
* 
* 
* 
��
VARIABLE_VALUE@Adam/behavior_model_1/rank_similarity_1/embedding_1/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE@Adam/behavior_model_1/rank_similarity_1/embedding_1/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameMbehavior_model_1/rank_similarity_1/embedding_1/embeddings/Read/ReadVariableOp#minkowski_1/rho/Read/ReadVariableOpUbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/w/Read/ReadVariableOp0exponential_similarity_1/tau/Read/ReadVariableOp2exponential_similarity_1/gamma/Read/ReadVariableOp1exponential_similarity_1/beta/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpTAdam/behavior_model_1/rank_similarity_1/embedding_1/embeddings/m/Read/ReadVariableOpTAdam/behavior_model_1/rank_similarity_1/embedding_1/embeddings/v/Read/ReadVariableOpConst_6*
Tin
2	*
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
GPU 2J 8� *(
f#R!
__inference__traced_save_220178
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename9behavior_model_1/rank_similarity_1/embedding_1/embeddingsminkowski_1/rhoAbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/wexponential_similarity_1/tauexponential_similarity_1/gammaexponential_similarity_1/beta	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcount@Adam/behavior_model_1/rank_similarity_1/embedding_1/embeddings/m@Adam/behavior_model_1/rank_similarity_1/embedding_1/embeddings/v*
Tin
2*
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
GPU 2J 8� *+
f&R$
"__inference__traced_restore_220239��
�<
�
#__inference_internal_grad_fn_220141
result_grads_0
result_grads_1
result_grads_2
result_grads_3S
Omul_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_broadcasttoK
Gmul_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_subK
Gpow_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_absR
Ndiv_no_nan_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_powR
Nsub_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_expanddimsO
Kpow_1_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_pow_1T
Pdiv_no_nan_1_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_sumO
Kmul_6_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_mul_1
identity

identity_1

identity_2�
mulMulOmul_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_broadcasttoGmul_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_sub*
T0*/
_output_shapes
:���������J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
powPowGpow_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_abspow/y:output:0*
T0*/
_output_shapes
:����������

div_no_nanDivNoNanNdiv_no_nan_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_powpow:z:0*
T0*/
_output_shapes
:���������_
mul_1Mulmul:z:0div_no_nan:z:0*
T0*/
_output_shapes
:���������J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
subSubNsub_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_expanddimssub/y:output:0*
T0*/
_output_shapes
:����������
pow_1PowKpow_1_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_pow_1sub:z:0*
T0*/
_output_shapes
:���������J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3a
addAddV2	pow_1:z:0add/y:output:0*
T0*/
_output_shapes
:���������`
truedivRealDiv	mul_1:z:0add:z:0*
T0*/
_output_shapes
:���������c
mul_2Mulresult_grads_0truediv:z:0*
T0*/
_output_shapes
:���������L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sub_1SubNsub_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_expanddimssub_1/y:output:0*
T0*/
_output_shapes
:����������
pow_2PowKpow_1_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_pow_1	sub_1:z:0*
T0*/
_output_shapes
:����������
mul_3MulNsub_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_expanddims	pow_2:z:0*
T0*/
_output_shapes
:���������L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3e
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*/
_output_shapes
:����������
	truediv_1RealDivNdiv_no_nan_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_pow	add_1:z:0*
T0*/
_output_shapes
:���������e
mul_4Mulresult_grads_0truediv_1:z:0*
T0*/
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
	truediv_2RealDivtruediv_2/x:output:0Nsub_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_expanddims*
T0*/
_output_shapes
:����������
div_no_nan_1DivNoNanKpow_1_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_pow_1Pdiv_no_nan_1_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_sum*
T0*/
_output_shapes
:���������g
mul_5Multruediv_2:z:0div_no_nan_1:z:0*
T0*/
_output_shapes
:���������L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
add_2AddV2Gpow_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_absadd_2/y:output:0*
T0*/
_output_shapes
:���������O
LogLog	add_2:z:0*
T0*/
_output_shapes
:����������
mul_6MulKmul_6_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_mul_1Log:y:0*
T0*/
_output_shapes
:���������`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
SumSum	mul_6:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(_
mul_7Mul	mul_5:z:0Sum:output:0*
T0*/
_output_shapes
:���������L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
pow_3PowNsub_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_expanddimspow_3/y:output:0*
T0*/
_output_shapes
:���������P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
	truediv_3RealDivtruediv_3/x:output:0	pow_3:z:0*
T0*/
_output_shapes
:����������
mul_8Multruediv_3:z:0Kpow_1_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_pow_1*
T0*/
_output_shapes
:���������L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
add_3AddV2Pdiv_no_nan_1_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_sumadd_3/y:output:0*
T0*/
_output_shapes
:���������Q
Log_1Log	add_3:z:0*
T0*/
_output_shapes
:���������\
mul_9Mul	mul_8:z:0	Log_1:y:0*
T0*/
_output_shapes
:���������\
sub_2Sub	mul_7:z:0	mul_9:z:0*
T0*/
_output_shapes
:���������b
mul_10Mulresult_grads_0	sub_2:z:0*
T0*/
_output_shapes
:���������t
SqueezeSqueeze
mul_10:z:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:���������[

Identity_1Identity	mul_4:z:0*
T0*/
_output_shapes
:���������^

Identity_2IdentitySqueeze:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:_ [
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_1:_[
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:���������
(
_user_specified_nameresult_grads_3:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:5	1
/
_output_shapes
:���������:5
1
/
_output_shapes
:���������:51
/
_output_shapes
:���������
�
�
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_218773

inputs	
rank_similarity_1_218747
rank_similarity_1_218749+
rank_similarity_1_218751:	�
rank_similarity_1_218753"
rank_similarity_1_218755: &
rank_similarity_1_218757:"
rank_similarity_1_218759: "
rank_similarity_1_218761: "
rank_similarity_1_218763: 
rank_similarity_1_218765
rank_similarity_1_218767
rank_similarity_1_218769
identity��)rank_similarity_1/StatefulPartitionedCall�
)rank_similarity_1/StatefulPartitionedCallStatefulPartitionedCallinputsrank_similarity_1_218747rank_similarity_1_218749rank_similarity_1_218751rank_similarity_1_218753rank_similarity_1_218755rank_similarity_1_218757rank_similarity_1_218759rank_similarity_1_218761rank_similarity_1_218763rank_similarity_1_218765rank_similarity_1_218767rank_similarity_1_218769*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������8*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218746�
IdentityIdentity2rank_similarity_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������8r
NoOpNoOp*^rank_similarity_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 2V
)rank_similarity_1/StatefulPartitionedCall)rank_similarity_1/StatefulPartitionedCall:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: 
�
�
1__inference_behavior_model_1_layer_call_fn_219225
inputs_8rank2_stimulus_set	
unknown
	unknown_0
	unknown_1:	�
	unknown_2
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_8rank2_stimulus_setunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������8*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219016s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������8`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 22
StatefulPartitionedCallStatefulPartitionedCall:g c
+
_output_shapes
:���������	
4
_user_specified_nameinputs/8rank2/stimulus_set: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: 
�i
�
M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219671
inputs_8rank2_stimulus_set	
gatherv2_indices
gatherv2_axis6
#embedding_1_embedding_lookup_219582:	�
packed_values_1>
4distance_based_1_minkowski_1_readvariableop_resource: N
@distance_based_1_minkowski_1_broadcastto_readvariableop_resource:O
Edistance_based_1_exponential_similarity_1_neg_readvariableop_resource: O
Edistance_based_1_exponential_similarity_1_pow_readvariableop_resource: O
Edistance_based_1_exponential_similarity_1_add_readvariableop_resource: 
gatherv2_1_indices	
mul_x
	truediv_y
identity��<distance_based_1/exponential_similarity_1/Neg/ReadVariableOp�<distance_based_1/exponential_similarity_1/Pow/ReadVariableOp�<distance_based_1/exponential_similarity_1/add/ReadVariableOp�7distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp�+distance_based_1/minkowski_1/ReadVariableOp�embedding_1/embedding_lookup�
GatherV2GatherV2inputs_8rank2_stimulus_setgatherv2_indicesgatherv2_axis*
Taxis0*
Tindices0*
Tparams0	*+
_output_shapes
:���������L

NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R r
NotEqualNotEqualGatherV2:output:0NotEqual/y:output:0*
T0	*+
_output_shapes
:����������
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_219582inputs_8rank2_stimulus_set*
Tindices0	*6
_class,
*(loc:@embedding_1/embedding_lookup/219582*/
_output_shapes
:���������	*
dtype0�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/219582*/
_output_shapes
:���������	�
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:���������	X
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
embedding_1/NotEqualNotEqualinputs_8rank2_stimulus_setembedding_1/NotEqual/y:output:0*
T0	*+
_output_shapes
:���������	J
packed/0Const*
_output_shapes
: *
dtype0*
value	B :`
packedPackpacked/0:output:0packed_values_1*
N*
T0*
_output_shapes
:Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitV0embedding_1/embedding_lookup/Identity_1:output:0packed:output:0split/split_dim:output:0*
T0*

Tlen0*J
_output_shapes8
6:���������:���������*
	num_split�
 distance_based_1/minkowski_1/subSubsplit:output:0split:output:1*
T0*/
_output_shapes
:���������v
"distance_based_1/minkowski_1/ShapeShape$distance_based_1/minkowski_1/sub:z:0*
T0*
_output_shapes
:z
0distance_based_1/minkowski_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
2distance_based_1/minkowski_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������|
2distance_based_1/minkowski_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*distance_based_1/minkowski_1/strided_sliceStridedSlice+distance_based_1/minkowski_1/Shape:output:09distance_based_1/minkowski_1/strided_slice/stack:output:0;distance_based_1/minkowski_1/strided_slice/stack_1:output:0;distance_based_1/minkowski_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:l
'distance_based_1/minkowski_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
!distance_based_1/minkowski_1/onesFill3distance_based_1/minkowski_1/strided_slice:output:00distance_based_1/minkowski_1/ones/Const:output:0*
T0*+
_output_shapes
:����������
+distance_based_1/minkowski_1/ReadVariableOpReadVariableOp4distance_based_1_minkowski_1_readvariableop_resource*
_output_shapes
: *
dtype0�
 distance_based_1/minkowski_1/mulMul3distance_based_1/minkowski_1/ReadVariableOp:value:0*distance_based_1/minkowski_1/ones:output:0*
T0*+
_output_shapes
:����������
7distance_based_1/minkowski_1/BroadcastTo/ReadVariableOpReadVariableOp@distance_based_1_minkowski_1_broadcastto_readvariableop_resource*
_output_shapes
:*
dtype0�
(distance_based_1/minkowski_1/BroadcastToBroadcastTo?distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp:value:0+distance_based_1/minkowski_1/Shape:output:0*
T0*/
_output_shapes
:����������
 distance_based_1/minkowski_1/AbsAbs$distance_based_1/minkowski_1/sub:z:0*
T0*/
_output_shapes
:���������v
+distance_based_1/minkowski_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
'distance_based_1/minkowski_1/ExpandDims
ExpandDims$distance_based_1/minkowski_1/mul:z:04distance_based_1/minkowski_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
 distance_based_1/minkowski_1/PowPow$distance_based_1/minkowski_1/Abs:y:00distance_based_1/minkowski_1/ExpandDims:output:0*
T0*/
_output_shapes
:����������
"distance_based_1/minkowski_1/Mul_1Mul$distance_based_1/minkowski_1/Pow:z:01distance_based_1/minkowski_1/BroadcastTo:output:0*
T0*/
_output_shapes
:���������}
2distance_based_1/minkowski_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
 distance_based_1/minkowski_1/SumSum&distance_based_1/minkowski_1/Mul_1:z:0;distance_based_1/minkowski_1/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(k
&distance_based_1/minkowski_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
$distance_based_1/minkowski_1/truedivRealDiv/distance_based_1/minkowski_1/truediv/x:output:00distance_based_1/minkowski_1/ExpandDims:output:0*
T0*/
_output_shapes
:����������
"distance_based_1/minkowski_1/Pow_1Pow)distance_based_1/minkowski_1/Sum:output:0(distance_based_1/minkowski_1/truediv:z:0*
T0*/
_output_shapes
:����������
%distance_based_1/minkowski_1/IdentityIdentity&distance_based_1/minkowski_1/Pow_1:z:0*
T0*/
_output_shapes
:����������
&distance_based_1/minkowski_1/IdentityN	IdentityN&distance_based_1/minkowski_1/Pow_1:z:0$distance_based_1/minkowski_1/sub:z:01distance_based_1/minkowski_1/BroadcastTo:output:0$distance_based_1/minkowski_1/mul:z:0*
T
2*,
_gradient_op_typeCustomGradient-219609*|
_output_shapesj
h:���������:���������:���������:����������
$distance_based_1/minkowski_1/SqueezeSqueeze/distance_based_1/minkowski_1/IdentityN:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

����������
<distance_based_1/exponential_similarity_1/Neg/ReadVariableOpReadVariableOpEdistance_based_1_exponential_similarity_1_neg_readvariableop_resource*
_output_shapes
: *
dtype0�
-distance_based_1/exponential_similarity_1/NegNegDdistance_based_1/exponential_similarity_1/Neg/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<distance_based_1/exponential_similarity_1/Pow/ReadVariableOpReadVariableOpEdistance_based_1_exponential_similarity_1_pow_readvariableop_resource*
_output_shapes
: *
dtype0�
-distance_based_1/exponential_similarity_1/PowPow-distance_based_1/minkowski_1/Squeeze:output:0Ddistance_based_1/exponential_similarity_1/Pow/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
-distance_based_1/exponential_similarity_1/mulMul1distance_based_1/exponential_similarity_1/Neg:y:01distance_based_1/exponential_similarity_1/Pow:z:0*
T0*+
_output_shapes
:����������
-distance_based_1/exponential_similarity_1/ExpExp1distance_based_1/exponential_similarity_1/mul:z:0*
T0*+
_output_shapes
:����������
<distance_based_1/exponential_similarity_1/add/ReadVariableOpReadVariableOpEdistance_based_1_exponential_similarity_1_add_readvariableop_resource*
_output_shapes
: *
dtype0�
-distance_based_1/exponential_similarity_1/addAddV21distance_based_1/exponential_similarity_1/Exp:y:0Ddistance_based_1/exponential_similarity_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������_
CastCastNotEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:����������
rank_sim_zero_out_nonpresentMul1distance_based_1/exponential_similarity_1/add:z:0Cast:y:0*
T0*+
_output_shapes
:����������

GatherV2_1GatherV2 rank_sim_zero_out_nonpresent:z:0gatherv2_1_indicesgatherv2_axis*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:���������8P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :u

ExpandDims
ExpandDimsCast:y:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������G
ConstConst*
_output_shapes
: *
dtype0*
value	B : �

GatherV2_2GatherV2ExpandDims:output:0Const:output:0gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:���������M
Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B :�
CumsumCumsumGatherV2_1:output:0Cumsum/axis:output:0*
T0*/
_output_shapes
:���������8*
reverse(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3u
MaximumMaximumGatherV2_1:output:0Maximum/y:output:0*
T0*/
_output_shapes
:���������8P
Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3r
	Maximum_1MaximumCumsum:out:0Maximum_1/y:output:0*
T0*/
_output_shapes
:���������8Q
LogLogMaximum:z:0*
T0*/
_output_shapes
:���������8U
Log_1LogMaximum_1:z:0*
T0*/
_output_shapes
:���������8X
subSubLog:y:0	Log_1:y:0*
T0*/
_output_shapes
:���������8T
mulMulmul_xsub:z:0*
T0*/
_output_shapes
:���������8W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :i
SumSummul:z:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:���������8N
ExpExpSum:output:0*
T0*+
_output_shapes
:���������8`
mul_1MulGatherV2_2:output:0Exp:y:0*
T0*+
_output_shapes
:���������8Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
Sum_1Sum	mul_1:z:0 Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    f
EqualEqualSum_1:output:0Equal/y:output:0*
T0*+
_output_shapes
:���������^
Cast_1Cast	Equal:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������_
truedivRealDiv
Cast_1:y:0	truediv_y*
T0*+
_output_shapes
:���������Z
addAddV2	mul_1:z:0truediv:z:0*
T0*+
_output_shapes
:���������8`
add_1AddV2Sum_1:output:0
Cast_1:y:0*
T0*+
_output_shapes
:���������^
	truediv_1RealDivadd:z:0	add_1:z:0*
T0*+
_output_shapes
:���������8`
IdentityIdentitytruediv_1:z:0^NoOp*
T0*+
_output_shapes
:���������8�
NoOpNoOp=^distance_based_1/exponential_similarity_1/Neg/ReadVariableOp=^distance_based_1/exponential_similarity_1/Pow/ReadVariableOp=^distance_based_1/exponential_similarity_1/add/ReadVariableOp8^distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp,^distance_based_1/minkowski_1/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 2|
<distance_based_1/exponential_similarity_1/Neg/ReadVariableOp<distance_based_1/exponential_similarity_1/Neg/ReadVariableOp2|
<distance_based_1/exponential_similarity_1/Pow/ReadVariableOp<distance_based_1/exponential_similarity_1/Pow/ReadVariableOp2|
<distance_based_1/exponential_similarity_1/add/ReadVariableOp<distance_based_1/exponential_similarity_1/add/ReadVariableOp2r
7distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp7distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp2Z
+distance_based_1/minkowski_1/ReadVariableOp+distance_based_1/minkowski_1/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:g c
+
_output_shapes
:���������	
4
_user_specified_nameinputs/8rank2/stimulus_set: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: 
�
�
$__inference_signature_wrapper_219167
rank2_stimulus_set	
unknown
	unknown_0
	unknown_1:	�
	unknown_2
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallrank2_stimulus_setunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������8*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_218642s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������8`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:���������	
-
_user_specified_name8rank2/stimulus_set: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: 
�i
�
M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219574
inputs_8rank2_stimulus_set	
gatherv2_indices
gatherv2_axis6
#embedding_1_embedding_lookup_219485:	�
packed_values_1>
4distance_based_1_minkowski_1_readvariableop_resource: N
@distance_based_1_minkowski_1_broadcastto_readvariableop_resource:O
Edistance_based_1_exponential_similarity_1_neg_readvariableop_resource: O
Edistance_based_1_exponential_similarity_1_pow_readvariableop_resource: O
Edistance_based_1_exponential_similarity_1_add_readvariableop_resource: 
gatherv2_1_indices	
mul_x
	truediv_y
identity��<distance_based_1/exponential_similarity_1/Neg/ReadVariableOp�<distance_based_1/exponential_similarity_1/Pow/ReadVariableOp�<distance_based_1/exponential_similarity_1/add/ReadVariableOp�7distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp�+distance_based_1/minkowski_1/ReadVariableOp�embedding_1/embedding_lookup�
GatherV2GatherV2inputs_8rank2_stimulus_setgatherv2_indicesgatherv2_axis*
Taxis0*
Tindices0*
Tparams0	*+
_output_shapes
:���������L

NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R r
NotEqualNotEqualGatherV2:output:0NotEqual/y:output:0*
T0	*+
_output_shapes
:����������
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_219485inputs_8rank2_stimulus_set*
Tindices0	*6
_class,
*(loc:@embedding_1/embedding_lookup/219485*/
_output_shapes
:���������	*
dtype0�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/219485*/
_output_shapes
:���������	�
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:���������	X
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
embedding_1/NotEqualNotEqualinputs_8rank2_stimulus_setembedding_1/NotEqual/y:output:0*
T0	*+
_output_shapes
:���������	J
packed/0Const*
_output_shapes
: *
dtype0*
value	B :`
packedPackpacked/0:output:0packed_values_1*
N*
T0*
_output_shapes
:Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitV0embedding_1/embedding_lookup/Identity_1:output:0packed:output:0split/split_dim:output:0*
T0*

Tlen0*J
_output_shapes8
6:���������:���������*
	num_split�
 distance_based_1/minkowski_1/subSubsplit:output:0split:output:1*
T0*/
_output_shapes
:���������v
"distance_based_1/minkowski_1/ShapeShape$distance_based_1/minkowski_1/sub:z:0*
T0*
_output_shapes
:z
0distance_based_1/minkowski_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
2distance_based_1/minkowski_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������|
2distance_based_1/minkowski_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*distance_based_1/minkowski_1/strided_sliceStridedSlice+distance_based_1/minkowski_1/Shape:output:09distance_based_1/minkowski_1/strided_slice/stack:output:0;distance_based_1/minkowski_1/strided_slice/stack_1:output:0;distance_based_1/minkowski_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:l
'distance_based_1/minkowski_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
!distance_based_1/minkowski_1/onesFill3distance_based_1/minkowski_1/strided_slice:output:00distance_based_1/minkowski_1/ones/Const:output:0*
T0*+
_output_shapes
:����������
+distance_based_1/minkowski_1/ReadVariableOpReadVariableOp4distance_based_1_minkowski_1_readvariableop_resource*
_output_shapes
: *
dtype0�
 distance_based_1/minkowski_1/mulMul3distance_based_1/minkowski_1/ReadVariableOp:value:0*distance_based_1/minkowski_1/ones:output:0*
T0*+
_output_shapes
:����������
7distance_based_1/minkowski_1/BroadcastTo/ReadVariableOpReadVariableOp@distance_based_1_minkowski_1_broadcastto_readvariableop_resource*
_output_shapes
:*
dtype0�
(distance_based_1/minkowski_1/BroadcastToBroadcastTo?distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp:value:0+distance_based_1/minkowski_1/Shape:output:0*
T0*/
_output_shapes
:����������
 distance_based_1/minkowski_1/AbsAbs$distance_based_1/minkowski_1/sub:z:0*
T0*/
_output_shapes
:���������v
+distance_based_1/minkowski_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
'distance_based_1/minkowski_1/ExpandDims
ExpandDims$distance_based_1/minkowski_1/mul:z:04distance_based_1/minkowski_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
 distance_based_1/minkowski_1/PowPow$distance_based_1/minkowski_1/Abs:y:00distance_based_1/minkowski_1/ExpandDims:output:0*
T0*/
_output_shapes
:����������
"distance_based_1/minkowski_1/Mul_1Mul$distance_based_1/minkowski_1/Pow:z:01distance_based_1/minkowski_1/BroadcastTo:output:0*
T0*/
_output_shapes
:���������}
2distance_based_1/minkowski_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
 distance_based_1/minkowski_1/SumSum&distance_based_1/minkowski_1/Mul_1:z:0;distance_based_1/minkowski_1/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(k
&distance_based_1/minkowski_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
$distance_based_1/minkowski_1/truedivRealDiv/distance_based_1/minkowski_1/truediv/x:output:00distance_based_1/minkowski_1/ExpandDims:output:0*
T0*/
_output_shapes
:����������
"distance_based_1/minkowski_1/Pow_1Pow)distance_based_1/minkowski_1/Sum:output:0(distance_based_1/minkowski_1/truediv:z:0*
T0*/
_output_shapes
:����������
%distance_based_1/minkowski_1/IdentityIdentity&distance_based_1/minkowski_1/Pow_1:z:0*
T0*/
_output_shapes
:����������
&distance_based_1/minkowski_1/IdentityN	IdentityN&distance_based_1/minkowski_1/Pow_1:z:0$distance_based_1/minkowski_1/sub:z:01distance_based_1/minkowski_1/BroadcastTo:output:0$distance_based_1/minkowski_1/mul:z:0*
T
2*,
_gradient_op_typeCustomGradient-219512*|
_output_shapesj
h:���������:���������:���������:����������
$distance_based_1/minkowski_1/SqueezeSqueeze/distance_based_1/minkowski_1/IdentityN:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

����������
<distance_based_1/exponential_similarity_1/Neg/ReadVariableOpReadVariableOpEdistance_based_1_exponential_similarity_1_neg_readvariableop_resource*
_output_shapes
: *
dtype0�
-distance_based_1/exponential_similarity_1/NegNegDdistance_based_1/exponential_similarity_1/Neg/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<distance_based_1/exponential_similarity_1/Pow/ReadVariableOpReadVariableOpEdistance_based_1_exponential_similarity_1_pow_readvariableop_resource*
_output_shapes
: *
dtype0�
-distance_based_1/exponential_similarity_1/PowPow-distance_based_1/minkowski_1/Squeeze:output:0Ddistance_based_1/exponential_similarity_1/Pow/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
-distance_based_1/exponential_similarity_1/mulMul1distance_based_1/exponential_similarity_1/Neg:y:01distance_based_1/exponential_similarity_1/Pow:z:0*
T0*+
_output_shapes
:����������
-distance_based_1/exponential_similarity_1/ExpExp1distance_based_1/exponential_similarity_1/mul:z:0*
T0*+
_output_shapes
:����������
<distance_based_1/exponential_similarity_1/add/ReadVariableOpReadVariableOpEdistance_based_1_exponential_similarity_1_add_readvariableop_resource*
_output_shapes
: *
dtype0�
-distance_based_1/exponential_similarity_1/addAddV21distance_based_1/exponential_similarity_1/Exp:y:0Ddistance_based_1/exponential_similarity_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������_
CastCastNotEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:����������
rank_sim_zero_out_nonpresentMul1distance_based_1/exponential_similarity_1/add:z:0Cast:y:0*
T0*+
_output_shapes
:����������

GatherV2_1GatherV2 rank_sim_zero_out_nonpresent:z:0gatherv2_1_indicesgatherv2_axis*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:���������8P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :u

ExpandDims
ExpandDimsCast:y:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������G
ConstConst*
_output_shapes
: *
dtype0*
value	B : �

GatherV2_2GatherV2ExpandDims:output:0Const:output:0gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:���������M
Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B :�
CumsumCumsumGatherV2_1:output:0Cumsum/axis:output:0*
T0*/
_output_shapes
:���������8*
reverse(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3u
MaximumMaximumGatherV2_1:output:0Maximum/y:output:0*
T0*/
_output_shapes
:���������8P
Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3r
	Maximum_1MaximumCumsum:out:0Maximum_1/y:output:0*
T0*/
_output_shapes
:���������8Q
LogLogMaximum:z:0*
T0*/
_output_shapes
:���������8U
Log_1LogMaximum_1:z:0*
T0*/
_output_shapes
:���������8X
subSubLog:y:0	Log_1:y:0*
T0*/
_output_shapes
:���������8T
mulMulmul_xsub:z:0*
T0*/
_output_shapes
:���������8W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :i
SumSummul:z:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:���������8N
ExpExpSum:output:0*
T0*+
_output_shapes
:���������8`
mul_1MulGatherV2_2:output:0Exp:y:0*
T0*+
_output_shapes
:���������8Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
Sum_1Sum	mul_1:z:0 Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    f
EqualEqualSum_1:output:0Equal/y:output:0*
T0*+
_output_shapes
:���������^
Cast_1Cast	Equal:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������_
truedivRealDiv
Cast_1:y:0	truediv_y*
T0*+
_output_shapes
:���������Z
addAddV2	mul_1:z:0truediv:z:0*
T0*+
_output_shapes
:���������8`
add_1AddV2Sum_1:output:0
Cast_1:y:0*
T0*+
_output_shapes
:���������^
	truediv_1RealDivadd:z:0	add_1:z:0*
T0*+
_output_shapes
:���������8`
IdentityIdentitytruediv_1:z:0^NoOp*
T0*+
_output_shapes
:���������8�
NoOpNoOp=^distance_based_1/exponential_similarity_1/Neg/ReadVariableOp=^distance_based_1/exponential_similarity_1/Pow/ReadVariableOp=^distance_based_1/exponential_similarity_1/add/ReadVariableOp8^distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp,^distance_based_1/minkowski_1/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 2|
<distance_based_1/exponential_similarity_1/Neg/ReadVariableOp<distance_based_1/exponential_similarity_1/Neg/ReadVariableOp2|
<distance_based_1/exponential_similarity_1/Pow/ReadVariableOp<distance_based_1/exponential_similarity_1/Pow/ReadVariableOp2|
<distance_based_1/exponential_similarity_1/add/ReadVariableOp<distance_based_1/exponential_similarity_1/add/ReadVariableOp2r
7distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp7distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp2Z
+distance_based_1/minkowski_1/ReadVariableOp+distance_based_1/minkowski_1/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:g c
+
_output_shapes
:���������	
4
_user_specified_nameinputs/8rank2/stimulus_set: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: 
�
�
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219130
rank2_stimulus_set	
rank_similarity_1_219104
rank_similarity_1_219106+
rank_similarity_1_219108:	�
rank_similarity_1_219110"
rank_similarity_1_219112: &
rank_similarity_1_219114:"
rank_similarity_1_219116: "
rank_similarity_1_219118: "
rank_similarity_1_219120: 
rank_similarity_1_219122
rank_similarity_1_219124
rank_similarity_1_219126
identity��)rank_similarity_1/StatefulPartitionedCall�
)rank_similarity_1/StatefulPartitionedCallStatefulPartitionedCallrank2_stimulus_setrank_similarity_1_219104rank_similarity_1_219106rank_similarity_1_219108rank_similarity_1_219110rank_similarity_1_219112rank_similarity_1_219114rank_similarity_1_219116rank_similarity_1_219118rank_similarity_1_219120rank_similarity_1_219122rank_similarity_1_219124rank_similarity_1_219126*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������8*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218929�
IdentityIdentity2rank_similarity_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������8r
NoOpNoOp*^rank_similarity_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 2V
)rank_similarity_1/StatefulPartitionedCall)rank_similarity_1/StatefulPartitionedCall:` \
+
_output_shapes
:���������	
-
_user_specified_name8rank2/stimulus_set: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: 
�
�
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219016

inputs	
rank_similarity_1_218990
rank_similarity_1_218992+
rank_similarity_1_218994:	�
rank_similarity_1_218996"
rank_similarity_1_218998: &
rank_similarity_1_219000:"
rank_similarity_1_219002: "
rank_similarity_1_219004: "
rank_similarity_1_219006: 
rank_similarity_1_219008
rank_similarity_1_219010
rank_similarity_1_219012
identity��)rank_similarity_1/StatefulPartitionedCall�
)rank_similarity_1/StatefulPartitionedCallStatefulPartitionedCallinputsrank_similarity_1_218990rank_similarity_1_218992rank_similarity_1_218994rank_similarity_1_218996rank_similarity_1_218998rank_similarity_1_219000rank_similarity_1_219002rank_similarity_1_219004rank_similarity_1_219006rank_similarity_1_219008rank_similarity_1_219010rank_similarity_1_219012*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������8*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218929�
IdentityIdentity2rank_similarity_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������8r
NoOpNoOp*^rank_similarity_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 2V
)rank_similarity_1/StatefulPartitionedCall)rank_similarity_1/StatefulPartitionedCall:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: 
�h
�
M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218746

inputs	
gatherv2_indices
gatherv2_axis6
#embedding_1_embedding_lookup_218657:	�
packed_values_1>
4distance_based_1_minkowski_1_readvariableop_resource: N
@distance_based_1_minkowski_1_broadcastto_readvariableop_resource:O
Edistance_based_1_exponential_similarity_1_neg_readvariableop_resource: O
Edistance_based_1_exponential_similarity_1_pow_readvariableop_resource: O
Edistance_based_1_exponential_similarity_1_add_readvariableop_resource: 
gatherv2_1_indices	
mul_x
	truediv_y
identity��<distance_based_1/exponential_similarity_1/Neg/ReadVariableOp�<distance_based_1/exponential_similarity_1/Pow/ReadVariableOp�<distance_based_1/exponential_similarity_1/add/ReadVariableOp�7distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp�+distance_based_1/minkowski_1/ReadVariableOp�embedding_1/embedding_lookup�
GatherV2GatherV2inputsgatherv2_indicesgatherv2_axis*
Taxis0*
Tindices0*
Tparams0	*+
_output_shapes
:���������L

NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R r
NotEqualNotEqualGatherV2:output:0NotEqual/y:output:0*
T0	*+
_output_shapes
:����������
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_218657inputs*
Tindices0	*6
_class,
*(loc:@embedding_1/embedding_lookup/218657*/
_output_shapes
:���������	*
dtype0�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/218657*/
_output_shapes
:���������	�
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:���������	X
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
embedding_1/NotEqualNotEqualinputsembedding_1/NotEqual/y:output:0*
T0	*+
_output_shapes
:���������	J
packed/0Const*
_output_shapes
: *
dtype0*
value	B :`
packedPackpacked/0:output:0packed_values_1*
N*
T0*
_output_shapes
:Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitV0embedding_1/embedding_lookup/Identity_1:output:0packed:output:0split/split_dim:output:0*
T0*

Tlen0*J
_output_shapes8
6:���������:���������*
	num_split�
 distance_based_1/minkowski_1/subSubsplit:output:0split:output:1*
T0*/
_output_shapes
:���������v
"distance_based_1/minkowski_1/ShapeShape$distance_based_1/minkowski_1/sub:z:0*
T0*
_output_shapes
:z
0distance_based_1/minkowski_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
2distance_based_1/minkowski_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������|
2distance_based_1/minkowski_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*distance_based_1/minkowski_1/strided_sliceStridedSlice+distance_based_1/minkowski_1/Shape:output:09distance_based_1/minkowski_1/strided_slice/stack:output:0;distance_based_1/minkowski_1/strided_slice/stack_1:output:0;distance_based_1/minkowski_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:l
'distance_based_1/minkowski_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
!distance_based_1/minkowski_1/onesFill3distance_based_1/minkowski_1/strided_slice:output:00distance_based_1/minkowski_1/ones/Const:output:0*
T0*+
_output_shapes
:����������
+distance_based_1/minkowski_1/ReadVariableOpReadVariableOp4distance_based_1_minkowski_1_readvariableop_resource*
_output_shapes
: *
dtype0�
 distance_based_1/minkowski_1/mulMul3distance_based_1/minkowski_1/ReadVariableOp:value:0*distance_based_1/minkowski_1/ones:output:0*
T0*+
_output_shapes
:����������
7distance_based_1/minkowski_1/BroadcastTo/ReadVariableOpReadVariableOp@distance_based_1_minkowski_1_broadcastto_readvariableop_resource*
_output_shapes
:*
dtype0�
(distance_based_1/minkowski_1/BroadcastToBroadcastTo?distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp:value:0+distance_based_1/minkowski_1/Shape:output:0*
T0*/
_output_shapes
:����������
 distance_based_1/minkowski_1/AbsAbs$distance_based_1/minkowski_1/sub:z:0*
T0*/
_output_shapes
:���������v
+distance_based_1/minkowski_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
'distance_based_1/minkowski_1/ExpandDims
ExpandDims$distance_based_1/minkowski_1/mul:z:04distance_based_1/minkowski_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
 distance_based_1/minkowski_1/PowPow$distance_based_1/minkowski_1/Abs:y:00distance_based_1/minkowski_1/ExpandDims:output:0*
T0*/
_output_shapes
:����������
"distance_based_1/minkowski_1/Mul_1Mul$distance_based_1/minkowski_1/Pow:z:01distance_based_1/minkowski_1/BroadcastTo:output:0*
T0*/
_output_shapes
:���������}
2distance_based_1/minkowski_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
 distance_based_1/minkowski_1/SumSum&distance_based_1/minkowski_1/Mul_1:z:0;distance_based_1/minkowski_1/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(k
&distance_based_1/minkowski_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
$distance_based_1/minkowski_1/truedivRealDiv/distance_based_1/minkowski_1/truediv/x:output:00distance_based_1/minkowski_1/ExpandDims:output:0*
T0*/
_output_shapes
:����������
"distance_based_1/minkowski_1/Pow_1Pow)distance_based_1/minkowski_1/Sum:output:0(distance_based_1/minkowski_1/truediv:z:0*
T0*/
_output_shapes
:����������
%distance_based_1/minkowski_1/IdentityIdentity&distance_based_1/minkowski_1/Pow_1:z:0*
T0*/
_output_shapes
:����������
&distance_based_1/minkowski_1/IdentityN	IdentityN&distance_based_1/minkowski_1/Pow_1:z:0$distance_based_1/minkowski_1/sub:z:01distance_based_1/minkowski_1/BroadcastTo:output:0$distance_based_1/minkowski_1/mul:z:0*
T
2*,
_gradient_op_typeCustomGradient-218684*|
_output_shapesj
h:���������:���������:���������:����������
$distance_based_1/minkowski_1/SqueezeSqueeze/distance_based_1/minkowski_1/IdentityN:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

����������
<distance_based_1/exponential_similarity_1/Neg/ReadVariableOpReadVariableOpEdistance_based_1_exponential_similarity_1_neg_readvariableop_resource*
_output_shapes
: *
dtype0�
-distance_based_1/exponential_similarity_1/NegNegDdistance_based_1/exponential_similarity_1/Neg/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<distance_based_1/exponential_similarity_1/Pow/ReadVariableOpReadVariableOpEdistance_based_1_exponential_similarity_1_pow_readvariableop_resource*
_output_shapes
: *
dtype0�
-distance_based_1/exponential_similarity_1/PowPow-distance_based_1/minkowski_1/Squeeze:output:0Ddistance_based_1/exponential_similarity_1/Pow/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
-distance_based_1/exponential_similarity_1/mulMul1distance_based_1/exponential_similarity_1/Neg:y:01distance_based_1/exponential_similarity_1/Pow:z:0*
T0*+
_output_shapes
:����������
-distance_based_1/exponential_similarity_1/ExpExp1distance_based_1/exponential_similarity_1/mul:z:0*
T0*+
_output_shapes
:����������
<distance_based_1/exponential_similarity_1/add/ReadVariableOpReadVariableOpEdistance_based_1_exponential_similarity_1_add_readvariableop_resource*
_output_shapes
: *
dtype0�
-distance_based_1/exponential_similarity_1/addAddV21distance_based_1/exponential_similarity_1/Exp:y:0Ddistance_based_1/exponential_similarity_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������_
CastCastNotEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:����������
rank_sim_zero_out_nonpresentMul1distance_based_1/exponential_similarity_1/add:z:0Cast:y:0*
T0*+
_output_shapes
:����������

GatherV2_1GatherV2 rank_sim_zero_out_nonpresent:z:0gatherv2_1_indicesgatherv2_axis*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:���������8P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :u

ExpandDims
ExpandDimsCast:y:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������G
ConstConst*
_output_shapes
: *
dtype0*
value	B : �

GatherV2_2GatherV2ExpandDims:output:0Const:output:0gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:���������M
Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B :�
CumsumCumsumGatherV2_1:output:0Cumsum/axis:output:0*
T0*/
_output_shapes
:���������8*
reverse(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3u
MaximumMaximumGatherV2_1:output:0Maximum/y:output:0*
T0*/
_output_shapes
:���������8P
Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3r
	Maximum_1MaximumCumsum:out:0Maximum_1/y:output:0*
T0*/
_output_shapes
:���������8Q
LogLogMaximum:z:0*
T0*/
_output_shapes
:���������8U
Log_1LogMaximum_1:z:0*
T0*/
_output_shapes
:���������8X
subSubLog:y:0	Log_1:y:0*
T0*/
_output_shapes
:���������8T
mulMulmul_xsub:z:0*
T0*/
_output_shapes
:���������8W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :i
SumSummul:z:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:���������8N
ExpExpSum:output:0*
T0*+
_output_shapes
:���������8`
mul_1MulGatherV2_2:output:0Exp:y:0*
T0*+
_output_shapes
:���������8Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
Sum_1Sum	mul_1:z:0 Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    f
EqualEqualSum_1:output:0Equal/y:output:0*
T0*+
_output_shapes
:���������^
Cast_1Cast	Equal:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������_
truedivRealDiv
Cast_1:y:0	truediv_y*
T0*+
_output_shapes
:���������Z
addAddV2	mul_1:z:0truediv:z:0*
T0*+
_output_shapes
:���������8`
add_1AddV2Sum_1:output:0
Cast_1:y:0*
T0*+
_output_shapes
:���������^
	truediv_1RealDivadd:z:0	add_1:z:0*
T0*+
_output_shapes
:���������8`
IdentityIdentitytruediv_1:z:0^NoOp*
T0*+
_output_shapes
:���������8�
NoOpNoOp=^distance_based_1/exponential_similarity_1/Neg/ReadVariableOp=^distance_based_1/exponential_similarity_1/Pow/ReadVariableOp=^distance_based_1/exponential_similarity_1/add/ReadVariableOp8^distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp,^distance_based_1/minkowski_1/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 2|
<distance_based_1/exponential_similarity_1/Neg/ReadVariableOp<distance_based_1/exponential_similarity_1/Neg/ReadVariableOp2|
<distance_based_1/exponential_similarity_1/Pow/ReadVariableOp<distance_based_1/exponential_similarity_1/Pow/ReadVariableOp2|
<distance_based_1/exponential_similarity_1/add/ReadVariableOp<distance_based_1/exponential_similarity_1/add/ReadVariableOp2r
7distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp7distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp2Z
+distance_based_1/minkowski_1/ReadVariableOp+distance_based_1/minkowski_1/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: 
��
�

L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219419
inputs_8rank2_stimulus_set	&
"rank_similarity_1_gatherv2_indices#
rank_similarity_1_gatherv2_axisH
5rank_similarity_1_embedding_1_embedding_lookup_219330:	�%
!rank_similarity_1_packed_values_1P
Frank_similarity_1_distance_based_1_minkowski_1_readvariableop_resource: `
Rrank_similarity_1_distance_based_1_minkowski_1_broadcastto_readvariableop_resource:a
Wrank_similarity_1_distance_based_1_exponential_similarity_1_neg_readvariableop_resource: a
Wrank_similarity_1_distance_based_1_exponential_similarity_1_pow_readvariableop_resource: a
Wrank_similarity_1_distance_based_1_exponential_similarity_1_add_readvariableop_resource: (
$rank_similarity_1_gatherv2_1_indices
rank_similarity_1_mul_x
rank_similarity_1_truediv_y
identity��Nrank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOp�Nrank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOp�Nrank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOp�Irank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp�=rank_similarity_1/distance_based_1/minkowski_1/ReadVariableOp�.rank_similarity_1/embedding_1/embedding_lookup�
rank_similarity_1/GatherV2GatherV2inputs_8rank2_stimulus_set"rank_similarity_1_gatherv2_indicesrank_similarity_1_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0	*+
_output_shapes
:���������^
rank_similarity_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
rank_similarity_1/NotEqualNotEqual#rank_similarity_1/GatherV2:output:0%rank_similarity_1/NotEqual/y:output:0*
T0	*+
_output_shapes
:����������
.rank_similarity_1/embedding_1/embedding_lookupResourceGather5rank_similarity_1_embedding_1_embedding_lookup_219330inputs_8rank2_stimulus_set*
Tindices0	*H
_class>
<:loc:@rank_similarity_1/embedding_1/embedding_lookup/219330*/
_output_shapes
:���������	*
dtype0�
7rank_similarity_1/embedding_1/embedding_lookup/IdentityIdentity7rank_similarity_1/embedding_1/embedding_lookup:output:0*
T0*H
_class>
<:loc:@rank_similarity_1/embedding_1/embedding_lookup/219330*/
_output_shapes
:���������	�
9rank_similarity_1/embedding_1/embedding_lookup/Identity_1Identity@rank_similarity_1/embedding_1/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:���������	j
(rank_similarity_1/embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
&rank_similarity_1/embedding_1/NotEqualNotEqualinputs_8rank2_stimulus_set1rank_similarity_1/embedding_1/NotEqual/y:output:0*
T0	*+
_output_shapes
:���������	\
rank_similarity_1/packed/0Const*
_output_shapes
: *
dtype0*
value	B :�
rank_similarity_1/packedPack#rank_similarity_1/packed/0:output:0!rank_similarity_1_packed_values_1*
N*
T0*
_output_shapes
:c
!rank_similarity_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
rank_similarity_1/splitSplitVBrank_similarity_1/embedding_1/embedding_lookup/Identity_1:output:0!rank_similarity_1/packed:output:0*rank_similarity_1/split/split_dim:output:0*
T0*

Tlen0*J
_output_shapes8
6:���������:���������*
	num_split�
2rank_similarity_1/distance_based_1/minkowski_1/subSub rank_similarity_1/split:output:0 rank_similarity_1/split:output:1*
T0*/
_output_shapes
:����������
4rank_similarity_1/distance_based_1/minkowski_1/ShapeShape6rank_similarity_1/distance_based_1/minkowski_1/sub:z:0*
T0*
_output_shapes
:�
Brank_similarity_1/distance_based_1/minkowski_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Drank_similarity_1/distance_based_1/minkowski_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
����������
Drank_similarity_1/distance_based_1/minkowski_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
<rank_similarity_1/distance_based_1/minkowski_1/strided_sliceStridedSlice=rank_similarity_1/distance_based_1/minkowski_1/Shape:output:0Krank_similarity_1/distance_based_1/minkowski_1/strided_slice/stack:output:0Mrank_similarity_1/distance_based_1/minkowski_1/strided_slice/stack_1:output:0Mrank_similarity_1/distance_based_1/minkowski_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:~
9rank_similarity_1/distance_based_1/minkowski_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
3rank_similarity_1/distance_based_1/minkowski_1/onesFillErank_similarity_1/distance_based_1/minkowski_1/strided_slice:output:0Brank_similarity_1/distance_based_1/minkowski_1/ones/Const:output:0*
T0*+
_output_shapes
:����������
=rank_similarity_1/distance_based_1/minkowski_1/ReadVariableOpReadVariableOpFrank_similarity_1_distance_based_1_minkowski_1_readvariableop_resource*
_output_shapes
: *
dtype0�
2rank_similarity_1/distance_based_1/minkowski_1/mulMulErank_similarity_1/distance_based_1/minkowski_1/ReadVariableOp:value:0<rank_similarity_1/distance_based_1/minkowski_1/ones:output:0*
T0*+
_output_shapes
:����������
Irank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOpReadVariableOpRrank_similarity_1_distance_based_1_minkowski_1_broadcastto_readvariableop_resource*
_output_shapes
:*
dtype0�
:rank_similarity_1/distance_based_1/minkowski_1/BroadcastToBroadcastToQrank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp:value:0=rank_similarity_1/distance_based_1/minkowski_1/Shape:output:0*
T0*/
_output_shapes
:����������
2rank_similarity_1/distance_based_1/minkowski_1/AbsAbs6rank_similarity_1/distance_based_1/minkowski_1/sub:z:0*
T0*/
_output_shapes
:����������
=rank_similarity_1/distance_based_1/minkowski_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
9rank_similarity_1/distance_based_1/minkowski_1/ExpandDims
ExpandDims6rank_similarity_1/distance_based_1/minkowski_1/mul:z:0Frank_similarity_1/distance_based_1/minkowski_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
2rank_similarity_1/distance_based_1/minkowski_1/PowPow6rank_similarity_1/distance_based_1/minkowski_1/Abs:y:0Brank_similarity_1/distance_based_1/minkowski_1/ExpandDims:output:0*
T0*/
_output_shapes
:����������
4rank_similarity_1/distance_based_1/minkowski_1/Mul_1Mul6rank_similarity_1/distance_based_1/minkowski_1/Pow:z:0Crank_similarity_1/distance_based_1/minkowski_1/BroadcastTo:output:0*
T0*/
_output_shapes
:����������
Drank_similarity_1/distance_based_1/minkowski_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
2rank_similarity_1/distance_based_1/minkowski_1/SumSum8rank_similarity_1/distance_based_1/minkowski_1/Mul_1:z:0Mrank_similarity_1/distance_based_1/minkowski_1/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(}
8rank_similarity_1/distance_based_1/minkowski_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
6rank_similarity_1/distance_based_1/minkowski_1/truedivRealDivArank_similarity_1/distance_based_1/minkowski_1/truediv/x:output:0Brank_similarity_1/distance_based_1/minkowski_1/ExpandDims:output:0*
T0*/
_output_shapes
:����������
4rank_similarity_1/distance_based_1/minkowski_1/Pow_1Pow;rank_similarity_1/distance_based_1/minkowski_1/Sum:output:0:rank_similarity_1/distance_based_1/minkowski_1/truediv:z:0*
T0*/
_output_shapes
:����������
7rank_similarity_1/distance_based_1/minkowski_1/IdentityIdentity8rank_similarity_1/distance_based_1/minkowski_1/Pow_1:z:0*
T0*/
_output_shapes
:����������
8rank_similarity_1/distance_based_1/minkowski_1/IdentityN	IdentityN8rank_similarity_1/distance_based_1/minkowski_1/Pow_1:z:06rank_similarity_1/distance_based_1/minkowski_1/sub:z:0Crank_similarity_1/distance_based_1/minkowski_1/BroadcastTo:output:06rank_similarity_1/distance_based_1/minkowski_1/mul:z:0*
T
2*,
_gradient_op_typeCustomGradient-219357*|
_output_shapesj
h:���������:���������:���������:����������
6rank_similarity_1/distance_based_1/minkowski_1/SqueezeSqueezeArank_similarity_1/distance_based_1/minkowski_1/IdentityN:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

����������
Nrank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOpReadVariableOpWrank_similarity_1_distance_based_1_exponential_similarity_1_neg_readvariableop_resource*
_output_shapes
: *
dtype0�
?rank_similarity_1/distance_based_1/exponential_similarity_1/NegNegVrank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOp:value:0*
T0*
_output_shapes
: �
Nrank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOpReadVariableOpWrank_similarity_1_distance_based_1_exponential_similarity_1_pow_readvariableop_resource*
_output_shapes
: *
dtype0�
?rank_similarity_1/distance_based_1/exponential_similarity_1/PowPow?rank_similarity_1/distance_based_1/minkowski_1/Squeeze:output:0Vrank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
?rank_similarity_1/distance_based_1/exponential_similarity_1/mulMulCrank_similarity_1/distance_based_1/exponential_similarity_1/Neg:y:0Crank_similarity_1/distance_based_1/exponential_similarity_1/Pow:z:0*
T0*+
_output_shapes
:����������
?rank_similarity_1/distance_based_1/exponential_similarity_1/ExpExpCrank_similarity_1/distance_based_1/exponential_similarity_1/mul:z:0*
T0*+
_output_shapes
:����������
Nrank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOpReadVariableOpWrank_similarity_1_distance_based_1_exponential_similarity_1_add_readvariableop_resource*
_output_shapes
: *
dtype0�
?rank_similarity_1/distance_based_1/exponential_similarity_1/addAddV2Crank_similarity_1/distance_based_1/exponential_similarity_1/Exp:y:0Vrank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
rank_similarity_1/CastCastrank_similarity_1/NotEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:����������
.rank_similarity_1/rank_sim_zero_out_nonpresentMulCrank_similarity_1/distance_based_1/exponential_similarity_1/add:z:0rank_similarity_1/Cast:y:0*
T0*+
_output_shapes
:����������
rank_similarity_1/GatherV2_1GatherV22rank_similarity_1/rank_sim_zero_out_nonpresent:z:0$rank_similarity_1_gatherv2_1_indicesrank_similarity_1_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:���������8b
 rank_similarity_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
rank_similarity_1/ExpandDims
ExpandDimsrank_similarity_1/Cast:y:0)rank_similarity_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Y
rank_similarity_1/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
rank_similarity_1/GatherV2_2GatherV2%rank_similarity_1/ExpandDims:output:0 rank_similarity_1/Const:output:0rank_similarity_1_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:���������_
rank_similarity_1/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B :�
rank_similarity_1/CumsumCumsum%rank_similarity_1/GatherV2_1:output:0&rank_similarity_1/Cumsum/axis:output:0*
T0*/
_output_shapes
:���������8*
reverse(`
rank_similarity_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
rank_similarity_1/MaximumMaximum%rank_similarity_1/GatherV2_1:output:0$rank_similarity_1/Maximum/y:output:0*
T0*/
_output_shapes
:���������8b
rank_similarity_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
rank_similarity_1/Maximum_1Maximumrank_similarity_1/Cumsum:out:0&rank_similarity_1/Maximum_1/y:output:0*
T0*/
_output_shapes
:���������8u
rank_similarity_1/LogLogrank_similarity_1/Maximum:z:0*
T0*/
_output_shapes
:���������8y
rank_similarity_1/Log_1Logrank_similarity_1/Maximum_1:z:0*
T0*/
_output_shapes
:���������8�
rank_similarity_1/subSubrank_similarity_1/Log:y:0rank_similarity_1/Log_1:y:0*
T0*/
_output_shapes
:���������8�
rank_similarity_1/mulMulrank_similarity_1_mul_xrank_similarity_1/sub:z:0*
T0*/
_output_shapes
:���������8i
'rank_similarity_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
rank_similarity_1/SumSumrank_similarity_1/mul:z:00rank_similarity_1/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:���������8r
rank_similarity_1/ExpExprank_similarity_1/Sum:output:0*
T0*+
_output_shapes
:���������8�
rank_similarity_1/mul_1Mul%rank_similarity_1/GatherV2_2:output:0rank_similarity_1/Exp:y:0*
T0*+
_output_shapes
:���������8k
)rank_similarity_1/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
rank_similarity_1/Sum_1Sumrank_similarity_1/mul_1:z:02rank_similarity_1/Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(^
rank_similarity_1/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
rank_similarity_1/EqualEqual rank_similarity_1/Sum_1:output:0"rank_similarity_1/Equal/y:output:0*
T0*+
_output_shapes
:����������
rank_similarity_1/Cast_1Castrank_similarity_1/Equal:z:0*

DstT0*

SrcT0
*+
_output_shapes
:����������
rank_similarity_1/truedivRealDivrank_similarity_1/Cast_1:y:0rank_similarity_1_truediv_y*
T0*+
_output_shapes
:����������
rank_similarity_1/addAddV2rank_similarity_1/mul_1:z:0rank_similarity_1/truediv:z:0*
T0*+
_output_shapes
:���������8�
rank_similarity_1/add_1AddV2 rank_similarity_1/Sum_1:output:0rank_similarity_1/Cast_1:y:0*
T0*+
_output_shapes
:����������
rank_similarity_1/truediv_1RealDivrank_similarity_1/add:z:0rank_similarity_1/add_1:z:0*
T0*+
_output_shapes
:���������8r
IdentityIdentityrank_similarity_1/truediv_1:z:0^NoOp*
T0*+
_output_shapes
:���������8�
NoOpNoOpO^rank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOpO^rank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOpO^rank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOpJ^rank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp>^rank_similarity_1/distance_based_1/minkowski_1/ReadVariableOp/^rank_similarity_1/embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 2�
Nrank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOpNrank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOp2�
Nrank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOpNrank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOp2�
Nrank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOpNrank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOp2�
Irank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOpIrank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp2~
=rank_similarity_1/distance_based_1/minkowski_1/ReadVariableOp=rank_similarity_1/distance_based_1/minkowski_1/ReadVariableOp2`
.rank_similarity_1/embedding_1/embedding_lookup.rank_similarity_1/embedding_1/embedding_lookup:g c
+
_output_shapes
:���������	
4
_user_specified_nameinputs/8rank2/stimulus_set: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: 
�
�
1__inference_behavior_model_1_layer_call_fn_219072
rank2_stimulus_set	
unknown
	unknown_0
	unknown_1:	�
	unknown_2
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallrank2_stimulus_setunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������8*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219016s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������8`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:���������	
-
_user_specified_name8rank2/stimulus_set: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: 
�
�
1__inference_behavior_model_1_layer_call_fn_219196
inputs_8rank2_stimulus_set	
unknown
	unknown_0
	unknown_1:	�
	unknown_2
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_8rank2_stimulus_setunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������8*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_218773s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������8`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 22
StatefulPartitionedCallStatefulPartitionedCall:g c
+
_output_shapes
:���������	
4
_user_specified_nameinputs/8rank2/stimulus_set: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: 
�8
�
#__inference_internal_grad_fn_220080
result_grads_0
result_grads_1
result_grads_2
result_grads_3B
>mul_rank_similarity_1_distance_based_1_minkowski_1_broadcastto:
6mul_rank_similarity_1_distance_based_1_minkowski_1_sub:
6pow_rank_similarity_1_distance_based_1_minkowski_1_absA
=div_no_nan_rank_similarity_1_distance_based_1_minkowski_1_powA
=sub_rank_similarity_1_distance_based_1_minkowski_1_expanddims>
:pow_1_rank_similarity_1_distance_based_1_minkowski_1_pow_1C
?div_no_nan_1_rank_similarity_1_distance_based_1_minkowski_1_sum>
:mul_6_rank_similarity_1_distance_based_1_minkowski_1_mul_1
identity

identity_1

identity_2�
mulMul>mul_rank_similarity_1_distance_based_1_minkowski_1_broadcastto6mul_rank_similarity_1_distance_based_1_minkowski_1_sub*
T0*/
_output_shapes
:���������J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
powPow6pow_rank_similarity_1_distance_based_1_minkowski_1_abspow/y:output:0*
T0*/
_output_shapes
:����������

div_no_nanDivNoNan=div_no_nan_rank_similarity_1_distance_based_1_minkowski_1_powpow:z:0*
T0*/
_output_shapes
:���������_
mul_1Mulmul:z:0div_no_nan:z:0*
T0*/
_output_shapes
:���������J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
subSub=sub_rank_similarity_1_distance_based_1_minkowski_1_expanddimssub/y:output:0*
T0*/
_output_shapes
:����������
pow_1Pow:pow_1_rank_similarity_1_distance_based_1_minkowski_1_pow_1sub:z:0*
T0*/
_output_shapes
:���������J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3a
addAddV2	pow_1:z:0add/y:output:0*
T0*/
_output_shapes
:���������`
truedivRealDiv	mul_1:z:0add:z:0*
T0*/
_output_shapes
:���������c
mul_2Mulresult_grads_0truediv:z:0*
T0*/
_output_shapes
:���������L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sub_1Sub=sub_rank_similarity_1_distance_based_1_minkowski_1_expanddimssub_1/y:output:0*
T0*/
_output_shapes
:����������
pow_2Pow:pow_1_rank_similarity_1_distance_based_1_minkowski_1_pow_1	sub_1:z:0*
T0*/
_output_shapes
:����������
mul_3Mul=sub_rank_similarity_1_distance_based_1_minkowski_1_expanddims	pow_2:z:0*
T0*/
_output_shapes
:���������L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3e
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*/
_output_shapes
:����������
	truediv_1RealDiv=div_no_nan_rank_similarity_1_distance_based_1_minkowski_1_pow	add_1:z:0*
T0*/
_output_shapes
:���������e
mul_4Mulresult_grads_0truediv_1:z:0*
T0*/
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
	truediv_2RealDivtruediv_2/x:output:0=sub_rank_similarity_1_distance_based_1_minkowski_1_expanddims*
T0*/
_output_shapes
:����������
div_no_nan_1DivNoNan:pow_1_rank_similarity_1_distance_based_1_minkowski_1_pow_1?div_no_nan_1_rank_similarity_1_distance_based_1_minkowski_1_sum*
T0*/
_output_shapes
:���������g
mul_5Multruediv_2:z:0div_no_nan_1:z:0*
T0*/
_output_shapes
:���������L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
add_2AddV26pow_rank_similarity_1_distance_based_1_minkowski_1_absadd_2/y:output:0*
T0*/
_output_shapes
:���������O
LogLog	add_2:z:0*
T0*/
_output_shapes
:����������
mul_6Mul:mul_6_rank_similarity_1_distance_based_1_minkowski_1_mul_1Log:y:0*
T0*/
_output_shapes
:���������`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
SumSum	mul_6:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(_
mul_7Mul	mul_5:z:0Sum:output:0*
T0*/
_output_shapes
:���������L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
pow_3Pow=sub_rank_similarity_1_distance_based_1_minkowski_1_expanddimspow_3/y:output:0*
T0*/
_output_shapes
:���������P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
	truediv_3RealDivtruediv_3/x:output:0	pow_3:z:0*
T0*/
_output_shapes
:����������
mul_8Multruediv_3:z:0:pow_1_rank_similarity_1_distance_based_1_minkowski_1_pow_1*
T0*/
_output_shapes
:���������L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
add_3AddV2?div_no_nan_1_rank_similarity_1_distance_based_1_minkowski_1_sumadd_3/y:output:0*
T0*/
_output_shapes
:���������Q
Log_1Log	add_3:z:0*
T0*/
_output_shapes
:���������\
mul_9Mul	mul_8:z:0	Log_1:y:0*
T0*/
_output_shapes
:���������\
sub_2Sub	mul_7:z:0	mul_9:z:0*
T0*/
_output_shapes
:���������b
mul_10Mulresult_grads_0	sub_2:z:0*
T0*/
_output_shapes
:���������t
SqueezeSqueeze
mul_10:z:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:���������[

Identity_1Identity	mul_4:z:0*
T0*/
_output_shapes
:���������^

Identity_2IdentitySqueeze:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:_ [
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_1:_[
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:���������
(
_user_specified_nameresult_grads_3:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:5	1
/
_output_shapes
:���������:5
1
/
_output_shapes
:���������:51
/
_output_shapes
:���������
��
�
!__inference__wrapped_model_218642
rank2_stimulus_set	7
3behavior_model_1_rank_similarity_1_gatherv2_indices4
0behavior_model_1_rank_similarity_1_gatherv2_axisY
Fbehavior_model_1_rank_similarity_1_embedding_1_embedding_lookup_218553:	�6
2behavior_model_1_rank_similarity_1_packed_values_1a
Wbehavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_readvariableop_resource: q
cbehavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_broadcastto_readvariableop_resource:r
hbehavior_model_1_rank_similarity_1_distance_based_1_exponential_similarity_1_neg_readvariableop_resource: r
hbehavior_model_1_rank_similarity_1_distance_based_1_exponential_similarity_1_pow_readvariableop_resource: r
hbehavior_model_1_rank_similarity_1_distance_based_1_exponential_similarity_1_add_readvariableop_resource: 9
5behavior_model_1_rank_similarity_1_gatherv2_1_indices,
(behavior_model_1_rank_similarity_1_mul_x0
,behavior_model_1_rank_similarity_1_truediv_y
identity��_behavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOp�_behavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOp�_behavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOp�Zbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp�Nbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/ReadVariableOp�?behavior_model_1/rank_similarity_1/embedding_1/embedding_lookup�
+behavior_model_1/rank_similarity_1/GatherV2GatherV2rank2_stimulus_set3behavior_model_1_rank_similarity_1_gatherv2_indices0behavior_model_1_rank_similarity_1_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0	*+
_output_shapes
:���������o
-behavior_model_1/rank_similarity_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
+behavior_model_1/rank_similarity_1/NotEqualNotEqual4behavior_model_1/rank_similarity_1/GatherV2:output:06behavior_model_1/rank_similarity_1/NotEqual/y:output:0*
T0	*+
_output_shapes
:����������
?behavior_model_1/rank_similarity_1/embedding_1/embedding_lookupResourceGatherFbehavior_model_1_rank_similarity_1_embedding_1_embedding_lookup_218553rank2_stimulus_set*
Tindices0	*Y
_classO
MKloc:@behavior_model_1/rank_similarity_1/embedding_1/embedding_lookup/218553*/
_output_shapes
:���������	*
dtype0�
Hbehavior_model_1/rank_similarity_1/embedding_1/embedding_lookup/IdentityIdentityHbehavior_model_1/rank_similarity_1/embedding_1/embedding_lookup:output:0*
T0*Y
_classO
MKloc:@behavior_model_1/rank_similarity_1/embedding_1/embedding_lookup/218553*/
_output_shapes
:���������	�
Jbehavior_model_1/rank_similarity_1/embedding_1/embedding_lookup/Identity_1IdentityQbehavior_model_1/rank_similarity_1/embedding_1/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:���������	{
9behavior_model_1/rank_similarity_1/embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
7behavior_model_1/rank_similarity_1/embedding_1/NotEqualNotEqualrank2_stimulus_setBbehavior_model_1/rank_similarity_1/embedding_1/NotEqual/y:output:0*
T0	*+
_output_shapes
:���������	m
+behavior_model_1/rank_similarity_1/packed/0Const*
_output_shapes
: *
dtype0*
value	B :�
)behavior_model_1/rank_similarity_1/packedPack4behavior_model_1/rank_similarity_1/packed/0:output:02behavior_model_1_rank_similarity_1_packed_values_1*
N*
T0*
_output_shapes
:t
2behavior_model_1/rank_similarity_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
(behavior_model_1/rank_similarity_1/splitSplitVSbehavior_model_1/rank_similarity_1/embedding_1/embedding_lookup/Identity_1:output:02behavior_model_1/rank_similarity_1/packed:output:0;behavior_model_1/rank_similarity_1/split/split_dim:output:0*
T0*

Tlen0*J
_output_shapes8
6:���������:���������*
	num_split�
Cbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/subSub1behavior_model_1/rank_similarity_1/split:output:01behavior_model_1/rank_similarity_1/split:output:1*
T0*/
_output_shapes
:����������
Ebehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/ShapeShapeGbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/sub:z:0*
T0*
_output_shapes
:�
Sbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Ubehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
����������
Ubehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Mbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/strided_sliceStridedSliceNbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Shape:output:0\behavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/strided_slice/stack:output:0^behavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/strided_slice/stack_1:output:0^behavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:�
Jbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Dbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/onesFillVbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/strided_slice:output:0Sbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/ones/Const:output:0*
T0*+
_output_shapes
:����������
Nbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/ReadVariableOpReadVariableOpWbehavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_readvariableop_resource*
_output_shapes
: *
dtype0�
Cbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/mulMulVbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/ReadVariableOp:value:0Mbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/ones:output:0*
T0*+
_output_shapes
:����������
Zbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOpReadVariableOpcbehavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_broadcastto_readvariableop_resource*
_output_shapes
:*
dtype0�
Kbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/BroadcastToBroadcastTobbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp:value:0Nbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Shape:output:0*
T0*/
_output_shapes
:����������
Cbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/AbsAbsGbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/sub:z:0*
T0*/
_output_shapes
:����������
Nbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Jbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/ExpandDims
ExpandDimsGbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/mul:z:0Wbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
Cbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/PowPowGbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Abs:y:0Sbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/ExpandDims:output:0*
T0*/
_output_shapes
:����������
Ebehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Mul_1MulGbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Pow:z:0Tbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/BroadcastTo:output:0*
T0*/
_output_shapes
:����������
Ubehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
Cbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/SumSumIbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Mul_1:z:0^behavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(�
Ibehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
Gbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/truedivRealDivRbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/truediv/x:output:0Sbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/ExpandDims:output:0*
T0*/
_output_shapes
:����������
Ebehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Pow_1PowLbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Sum:output:0Kbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/truediv:z:0*
T0*/
_output_shapes
:����������
Hbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/IdentityIdentityIbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Pow_1:z:0*
T0*/
_output_shapes
:����������
Ibehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/IdentityN	IdentityNIbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Pow_1:z:0Gbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/sub:z:0Tbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/BroadcastTo:output:0Gbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/mul:z:0*
T
2*,
_gradient_op_typeCustomGradient-218580*|
_output_shapesj
h:���������:���������:���������:����������
Gbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/SqueezeSqueezeRbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/IdentityN:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

����������
_behavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOpReadVariableOphbehavior_model_1_rank_similarity_1_distance_based_1_exponential_similarity_1_neg_readvariableop_resource*
_output_shapes
: *
dtype0�
Pbehavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/NegNeggbehavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOp:value:0*
T0*
_output_shapes
: �
_behavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOpReadVariableOphbehavior_model_1_rank_similarity_1_distance_based_1_exponential_similarity_1_pow_readvariableop_resource*
_output_shapes
: *
dtype0�
Pbehavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/PowPowPbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Squeeze:output:0gbehavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
Pbehavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/mulMulTbehavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/Neg:y:0Tbehavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/Pow:z:0*
T0*+
_output_shapes
:����������
Pbehavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/ExpExpTbehavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/mul:z:0*
T0*+
_output_shapes
:����������
_behavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOpReadVariableOphbehavior_model_1_rank_similarity_1_distance_based_1_exponential_similarity_1_add_readvariableop_resource*
_output_shapes
: *
dtype0�
Pbehavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/addAddV2Tbehavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/Exp:y:0gbehavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
'behavior_model_1/rank_similarity_1/CastCast/behavior_model_1/rank_similarity_1/NotEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:����������
?behavior_model_1/rank_similarity_1/rank_sim_zero_out_nonpresentMulTbehavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/add:z:0+behavior_model_1/rank_similarity_1/Cast:y:0*
T0*+
_output_shapes
:����������
-behavior_model_1/rank_similarity_1/GatherV2_1GatherV2Cbehavior_model_1/rank_similarity_1/rank_sim_zero_out_nonpresent:z:05behavior_model_1_rank_similarity_1_gatherv2_1_indices0behavior_model_1_rank_similarity_1_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:���������8s
1behavior_model_1/rank_similarity_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
-behavior_model_1/rank_similarity_1/ExpandDims
ExpandDims+behavior_model_1/rank_similarity_1/Cast:y:0:behavior_model_1/rank_similarity_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������j
(behavior_model_1/rank_similarity_1/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
-behavior_model_1/rank_similarity_1/GatherV2_2GatherV26behavior_model_1/rank_similarity_1/ExpandDims:output:01behavior_model_1/rank_similarity_1/Const:output:00behavior_model_1_rank_similarity_1_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:���������p
.behavior_model_1/rank_similarity_1/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B :�
)behavior_model_1/rank_similarity_1/CumsumCumsum6behavior_model_1/rank_similarity_1/GatherV2_1:output:07behavior_model_1/rank_similarity_1/Cumsum/axis:output:0*
T0*/
_output_shapes
:���������8*
reverse(q
,behavior_model_1/rank_similarity_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
*behavior_model_1/rank_similarity_1/MaximumMaximum6behavior_model_1/rank_similarity_1/GatherV2_1:output:05behavior_model_1/rank_similarity_1/Maximum/y:output:0*
T0*/
_output_shapes
:���������8s
.behavior_model_1/rank_similarity_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
,behavior_model_1/rank_similarity_1/Maximum_1Maximum/behavior_model_1/rank_similarity_1/Cumsum:out:07behavior_model_1/rank_similarity_1/Maximum_1/y:output:0*
T0*/
_output_shapes
:���������8�
&behavior_model_1/rank_similarity_1/LogLog.behavior_model_1/rank_similarity_1/Maximum:z:0*
T0*/
_output_shapes
:���������8�
(behavior_model_1/rank_similarity_1/Log_1Log0behavior_model_1/rank_similarity_1/Maximum_1:z:0*
T0*/
_output_shapes
:���������8�
&behavior_model_1/rank_similarity_1/subSub*behavior_model_1/rank_similarity_1/Log:y:0,behavior_model_1/rank_similarity_1/Log_1:y:0*
T0*/
_output_shapes
:���������8�
&behavior_model_1/rank_similarity_1/mulMul(behavior_model_1_rank_similarity_1_mul_x*behavior_model_1/rank_similarity_1/sub:z:0*
T0*/
_output_shapes
:���������8z
8behavior_model_1/rank_similarity_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
&behavior_model_1/rank_similarity_1/SumSum*behavior_model_1/rank_similarity_1/mul:z:0Abehavior_model_1/rank_similarity_1/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:���������8�
&behavior_model_1/rank_similarity_1/ExpExp/behavior_model_1/rank_similarity_1/Sum:output:0*
T0*+
_output_shapes
:���������8�
(behavior_model_1/rank_similarity_1/mul_1Mul6behavior_model_1/rank_similarity_1/GatherV2_2:output:0*behavior_model_1/rank_similarity_1/Exp:y:0*
T0*+
_output_shapes
:���������8|
:behavior_model_1/rank_similarity_1/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
(behavior_model_1/rank_similarity_1/Sum_1Sum,behavior_model_1/rank_similarity_1/mul_1:z:0Cbehavior_model_1/rank_similarity_1/Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(o
*behavior_model_1/rank_similarity_1/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
(behavior_model_1/rank_similarity_1/EqualEqual1behavior_model_1/rank_similarity_1/Sum_1:output:03behavior_model_1/rank_similarity_1/Equal/y:output:0*
T0*+
_output_shapes
:����������
)behavior_model_1/rank_similarity_1/Cast_1Cast,behavior_model_1/rank_similarity_1/Equal:z:0*

DstT0*

SrcT0
*+
_output_shapes
:����������
*behavior_model_1/rank_similarity_1/truedivRealDiv-behavior_model_1/rank_similarity_1/Cast_1:y:0,behavior_model_1_rank_similarity_1_truediv_y*
T0*+
_output_shapes
:����������
&behavior_model_1/rank_similarity_1/addAddV2,behavior_model_1/rank_similarity_1/mul_1:z:0.behavior_model_1/rank_similarity_1/truediv:z:0*
T0*+
_output_shapes
:���������8�
(behavior_model_1/rank_similarity_1/add_1AddV21behavior_model_1/rank_similarity_1/Sum_1:output:0-behavior_model_1/rank_similarity_1/Cast_1:y:0*
T0*+
_output_shapes
:����������
,behavior_model_1/rank_similarity_1/truediv_1RealDiv*behavior_model_1/rank_similarity_1/add:z:0,behavior_model_1/rank_similarity_1/add_1:z:0*
T0*+
_output_shapes
:���������8�
IdentityIdentity0behavior_model_1/rank_similarity_1/truediv_1:z:0^NoOp*
T0*+
_output_shapes
:���������8�
NoOpNoOp`^behavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOp`^behavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOp`^behavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOp[^behavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOpO^behavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/ReadVariableOp@^behavior_model_1/rank_similarity_1/embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 2�
_behavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOp_behavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOp2�
_behavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOp_behavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOp2�
_behavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOp_behavior_model_1/rank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOp2�
Zbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOpZbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp2�
Nbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/ReadVariableOpNbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/ReadVariableOp2�
?behavior_model_1/rank_similarity_1/embedding_1/embedding_lookup?behavior_model_1/rank_similarity_1/embedding_1/embedding_lookup:` \
+
_output_shapes
:���������	
-
_user_specified_name8rank2/stimulus_set: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: 
�h
�
M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218929

inputs	
gatherv2_indices
gatherv2_axis6
#embedding_1_embedding_lookup_218840:	�
packed_values_1>
4distance_based_1_minkowski_1_readvariableop_resource: N
@distance_based_1_minkowski_1_broadcastto_readvariableop_resource:O
Edistance_based_1_exponential_similarity_1_neg_readvariableop_resource: O
Edistance_based_1_exponential_similarity_1_pow_readvariableop_resource: O
Edistance_based_1_exponential_similarity_1_add_readvariableop_resource: 
gatherv2_1_indices	
mul_x
	truediv_y
identity��<distance_based_1/exponential_similarity_1/Neg/ReadVariableOp�<distance_based_1/exponential_similarity_1/Pow/ReadVariableOp�<distance_based_1/exponential_similarity_1/add/ReadVariableOp�7distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp�+distance_based_1/minkowski_1/ReadVariableOp�embedding_1/embedding_lookup�
GatherV2GatherV2inputsgatherv2_indicesgatherv2_axis*
Taxis0*
Tindices0*
Tparams0	*+
_output_shapes
:���������L

NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R r
NotEqualNotEqualGatherV2:output:0NotEqual/y:output:0*
T0	*+
_output_shapes
:����������
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_218840inputs*
Tindices0	*6
_class,
*(loc:@embedding_1/embedding_lookup/218840*/
_output_shapes
:���������	*
dtype0�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/218840*/
_output_shapes
:���������	�
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:���������	X
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 
embedding_1/NotEqualNotEqualinputsembedding_1/NotEqual/y:output:0*
T0	*+
_output_shapes
:���������	J
packed/0Const*
_output_shapes
: *
dtype0*
value	B :`
packedPackpacked/0:output:0packed_values_1*
N*
T0*
_output_shapes
:Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitV0embedding_1/embedding_lookup/Identity_1:output:0packed:output:0split/split_dim:output:0*
T0*

Tlen0*J
_output_shapes8
6:���������:���������*
	num_split�
 distance_based_1/minkowski_1/subSubsplit:output:0split:output:1*
T0*/
_output_shapes
:���������v
"distance_based_1/minkowski_1/ShapeShape$distance_based_1/minkowski_1/sub:z:0*
T0*
_output_shapes
:z
0distance_based_1/minkowski_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
2distance_based_1/minkowski_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������|
2distance_based_1/minkowski_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
*distance_based_1/minkowski_1/strided_sliceStridedSlice+distance_based_1/minkowski_1/Shape:output:09distance_based_1/minkowski_1/strided_slice/stack:output:0;distance_based_1/minkowski_1/strided_slice/stack_1:output:0;distance_based_1/minkowski_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:l
'distance_based_1/minkowski_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
!distance_based_1/minkowski_1/onesFill3distance_based_1/minkowski_1/strided_slice:output:00distance_based_1/minkowski_1/ones/Const:output:0*
T0*+
_output_shapes
:����������
+distance_based_1/minkowski_1/ReadVariableOpReadVariableOp4distance_based_1_minkowski_1_readvariableop_resource*
_output_shapes
: *
dtype0�
 distance_based_1/minkowski_1/mulMul3distance_based_1/minkowski_1/ReadVariableOp:value:0*distance_based_1/minkowski_1/ones:output:0*
T0*+
_output_shapes
:����������
7distance_based_1/minkowski_1/BroadcastTo/ReadVariableOpReadVariableOp@distance_based_1_minkowski_1_broadcastto_readvariableop_resource*
_output_shapes
:*
dtype0�
(distance_based_1/minkowski_1/BroadcastToBroadcastTo?distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp:value:0+distance_based_1/minkowski_1/Shape:output:0*
T0*/
_output_shapes
:����������
 distance_based_1/minkowski_1/AbsAbs$distance_based_1/minkowski_1/sub:z:0*
T0*/
_output_shapes
:���������v
+distance_based_1/minkowski_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
'distance_based_1/minkowski_1/ExpandDims
ExpandDims$distance_based_1/minkowski_1/mul:z:04distance_based_1/minkowski_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
 distance_based_1/minkowski_1/PowPow$distance_based_1/minkowski_1/Abs:y:00distance_based_1/minkowski_1/ExpandDims:output:0*
T0*/
_output_shapes
:����������
"distance_based_1/minkowski_1/Mul_1Mul$distance_based_1/minkowski_1/Pow:z:01distance_based_1/minkowski_1/BroadcastTo:output:0*
T0*/
_output_shapes
:���������}
2distance_based_1/minkowski_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
 distance_based_1/minkowski_1/SumSum&distance_based_1/minkowski_1/Mul_1:z:0;distance_based_1/minkowski_1/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(k
&distance_based_1/minkowski_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
$distance_based_1/minkowski_1/truedivRealDiv/distance_based_1/minkowski_1/truediv/x:output:00distance_based_1/minkowski_1/ExpandDims:output:0*
T0*/
_output_shapes
:����������
"distance_based_1/minkowski_1/Pow_1Pow)distance_based_1/minkowski_1/Sum:output:0(distance_based_1/minkowski_1/truediv:z:0*
T0*/
_output_shapes
:����������
%distance_based_1/minkowski_1/IdentityIdentity&distance_based_1/minkowski_1/Pow_1:z:0*
T0*/
_output_shapes
:����������
&distance_based_1/minkowski_1/IdentityN	IdentityN&distance_based_1/minkowski_1/Pow_1:z:0$distance_based_1/minkowski_1/sub:z:01distance_based_1/minkowski_1/BroadcastTo:output:0$distance_based_1/minkowski_1/mul:z:0*
T
2*,
_gradient_op_typeCustomGradient-218867*|
_output_shapesj
h:���������:���������:���������:����������
$distance_based_1/minkowski_1/SqueezeSqueeze/distance_based_1/minkowski_1/IdentityN:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

����������
<distance_based_1/exponential_similarity_1/Neg/ReadVariableOpReadVariableOpEdistance_based_1_exponential_similarity_1_neg_readvariableop_resource*
_output_shapes
: *
dtype0�
-distance_based_1/exponential_similarity_1/NegNegDdistance_based_1/exponential_similarity_1/Neg/ReadVariableOp:value:0*
T0*
_output_shapes
: �
<distance_based_1/exponential_similarity_1/Pow/ReadVariableOpReadVariableOpEdistance_based_1_exponential_similarity_1_pow_readvariableop_resource*
_output_shapes
: *
dtype0�
-distance_based_1/exponential_similarity_1/PowPow-distance_based_1/minkowski_1/Squeeze:output:0Ddistance_based_1/exponential_similarity_1/Pow/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
-distance_based_1/exponential_similarity_1/mulMul1distance_based_1/exponential_similarity_1/Neg:y:01distance_based_1/exponential_similarity_1/Pow:z:0*
T0*+
_output_shapes
:����������
-distance_based_1/exponential_similarity_1/ExpExp1distance_based_1/exponential_similarity_1/mul:z:0*
T0*+
_output_shapes
:����������
<distance_based_1/exponential_similarity_1/add/ReadVariableOpReadVariableOpEdistance_based_1_exponential_similarity_1_add_readvariableop_resource*
_output_shapes
: *
dtype0�
-distance_based_1/exponential_similarity_1/addAddV21distance_based_1/exponential_similarity_1/Exp:y:0Ddistance_based_1/exponential_similarity_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������_
CastCastNotEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:����������
rank_sim_zero_out_nonpresentMul1distance_based_1/exponential_similarity_1/add:z:0Cast:y:0*
T0*+
_output_shapes
:����������

GatherV2_1GatherV2 rank_sim_zero_out_nonpresent:z:0gatherv2_1_indicesgatherv2_axis*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:���������8P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :u

ExpandDims
ExpandDimsCast:y:0ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������G
ConstConst*
_output_shapes
: *
dtype0*
value	B : �

GatherV2_2GatherV2ExpandDims:output:0Const:output:0gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:���������M
Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B :�
CumsumCumsumGatherV2_1:output:0Cumsum/axis:output:0*
T0*/
_output_shapes
:���������8*
reverse(N
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3u
MaximumMaximumGatherV2_1:output:0Maximum/y:output:0*
T0*/
_output_shapes
:���������8P
Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3r
	Maximum_1MaximumCumsum:out:0Maximum_1/y:output:0*
T0*/
_output_shapes
:���������8Q
LogLogMaximum:z:0*
T0*/
_output_shapes
:���������8U
Log_1LogMaximum_1:z:0*
T0*/
_output_shapes
:���������8X
subSubLog:y:0	Log_1:y:0*
T0*/
_output_shapes
:���������8T
mulMulmul_xsub:z:0*
T0*/
_output_shapes
:���������8W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :i
SumSummul:z:0Sum/reduction_indices:output:0*
T0*+
_output_shapes
:���������8N
ExpExpSum:output:0*
T0*+
_output_shapes
:���������8`
mul_1MulGatherV2_2:output:0Exp:y:0*
T0*+
_output_shapes
:���������8Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
Sum_1Sum	mul_1:z:0 Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(L
Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    f
EqualEqualSum_1:output:0Equal/y:output:0*
T0*+
_output_shapes
:���������^
Cast_1Cast	Equal:z:0*

DstT0*

SrcT0
*+
_output_shapes
:���������_
truedivRealDiv
Cast_1:y:0	truediv_y*
T0*+
_output_shapes
:���������Z
addAddV2	mul_1:z:0truediv:z:0*
T0*+
_output_shapes
:���������8`
add_1AddV2Sum_1:output:0
Cast_1:y:0*
T0*+
_output_shapes
:���������^
	truediv_1RealDivadd:z:0	add_1:z:0*
T0*+
_output_shapes
:���������8`
IdentityIdentitytruediv_1:z:0^NoOp*
T0*+
_output_shapes
:���������8�
NoOpNoOp=^distance_based_1/exponential_similarity_1/Neg/ReadVariableOp=^distance_based_1/exponential_similarity_1/Pow/ReadVariableOp=^distance_based_1/exponential_similarity_1/add/ReadVariableOp8^distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp,^distance_based_1/minkowski_1/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 2|
<distance_based_1/exponential_similarity_1/Neg/ReadVariableOp<distance_based_1/exponential_similarity_1/Neg/ReadVariableOp2|
<distance_based_1/exponential_similarity_1/Pow/ReadVariableOp<distance_based_1/exponential_similarity_1/Pow/ReadVariableOp2|
<distance_based_1/exponential_similarity_1/add/ReadVariableOp<distance_based_1/exponential_similarity_1/add/ReadVariableOp2r
7distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp7distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp2Z
+distance_based_1/minkowski_1/ReadVariableOp+distance_based_1/minkowski_1/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:S O
+
_output_shapes
:���������	
 
_user_specified_nameinputs: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: 
�
�
2__inference_rank_similarity_1_layer_call_fn_219477
inputs_8rank2_stimulus_set	
unknown
	unknown_0
	unknown_1:	�
	unknown_2
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_8rank2_stimulus_setunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������8*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218929s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������8`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 22
StatefulPartitionedCallStatefulPartitionedCall:g c
+
_output_shapes
:���������	
4
_user_specified_nameinputs/8rank2/stimulus_set: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: 
�4
�
#__inference_internal_grad_fn_219775
result_grads_0
result_grads_1
result_grads_2
result_grads_30
,mul_distance_based_1_minkowski_1_broadcastto(
$mul_distance_based_1_minkowski_1_sub(
$pow_distance_based_1_minkowski_1_abs/
+div_no_nan_distance_based_1_minkowski_1_pow/
+sub_distance_based_1_minkowski_1_expanddims,
(pow_1_distance_based_1_minkowski_1_pow_11
-div_no_nan_1_distance_based_1_minkowski_1_sum,
(mul_6_distance_based_1_minkowski_1_mul_1
identity

identity_1

identity_2�
mulMul,mul_distance_based_1_minkowski_1_broadcastto$mul_distance_based_1_minkowski_1_sub*
T0*/
_output_shapes
:���������J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
powPow$pow_distance_based_1_minkowski_1_abspow/y:output:0*
T0*/
_output_shapes
:����������

div_no_nanDivNoNan+div_no_nan_distance_based_1_minkowski_1_powpow:z:0*
T0*/
_output_shapes
:���������_
mul_1Mulmul:z:0div_no_nan:z:0*
T0*/
_output_shapes
:���������J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
subSub+sub_distance_based_1_minkowski_1_expanddimssub/y:output:0*
T0*/
_output_shapes
:���������y
pow_1Pow(pow_1_distance_based_1_minkowski_1_pow_1sub:z:0*
T0*/
_output_shapes
:���������J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3a
addAddV2	pow_1:z:0add/y:output:0*
T0*/
_output_shapes
:���������`
truedivRealDiv	mul_1:z:0add:z:0*
T0*/
_output_shapes
:���������c
mul_2Mulresult_grads_0truediv:z:0*
T0*/
_output_shapes
:���������L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sub_1Sub+sub_distance_based_1_minkowski_1_expanddimssub_1/y:output:0*
T0*/
_output_shapes
:���������{
pow_2Pow(pow_1_distance_based_1_minkowski_1_pow_1	sub_1:z:0*
T0*/
_output_shapes
:���������~
mul_3Mul+sub_distance_based_1_minkowski_1_expanddims	pow_2:z:0*
T0*/
_output_shapes
:���������L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3e
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*/
_output_shapes
:����������
	truediv_1RealDiv+div_no_nan_distance_based_1_minkowski_1_pow	add_1:z:0*
T0*/
_output_shapes
:���������e
mul_4Mulresult_grads_0truediv_1:z:0*
T0*/
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
	truediv_2RealDivtruediv_2/x:output:0+sub_distance_based_1_minkowski_1_expanddims*
T0*/
_output_shapes
:����������
div_no_nan_1DivNoNan(pow_1_distance_based_1_minkowski_1_pow_1-div_no_nan_1_distance_based_1_minkowski_1_sum*
T0*/
_output_shapes
:���������g
mul_5Multruediv_2:z:0div_no_nan_1:z:0*
T0*/
_output_shapes
:���������L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
add_2AddV2$pow_distance_based_1_minkowski_1_absadd_2/y:output:0*
T0*/
_output_shapes
:���������O
LogLog	add_2:z:0*
T0*/
_output_shapes
:���������y
mul_6Mul(mul_6_distance_based_1_minkowski_1_mul_1Log:y:0*
T0*/
_output_shapes
:���������`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
SumSum	mul_6:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(_
mul_7Mul	mul_5:z:0Sum:output:0*
T0*/
_output_shapes
:���������L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
pow_3Pow+sub_distance_based_1_minkowski_1_expanddimspow_3/y:output:0*
T0*/
_output_shapes
:���������P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
	truediv_3RealDivtruediv_3/x:output:0	pow_3:z:0*
T0*/
_output_shapes
:���������
mul_8Multruediv_3:z:0(pow_1_distance_based_1_minkowski_1_pow_1*
T0*/
_output_shapes
:���������L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
add_3AddV2-div_no_nan_1_distance_based_1_minkowski_1_sumadd_3/y:output:0*
T0*/
_output_shapes
:���������Q
Log_1Log	add_3:z:0*
T0*/
_output_shapes
:���������\
mul_9Mul	mul_8:z:0	Log_1:y:0*
T0*/
_output_shapes
:���������\
sub_2Sub	mul_7:z:0	mul_9:z:0*
T0*/
_output_shapes
:���������b
mul_10Mulresult_grads_0	sub_2:z:0*
T0*/
_output_shapes
:���������t
SqueezeSqueeze
mul_10:z:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:���������[

Identity_1Identity	mul_4:z:0*
T0*/
_output_shapes
:���������^

Identity_2IdentitySqueeze:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:_ [
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_1:_[
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:���������
(
_user_specified_nameresult_grads_3:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:5	1
/
_output_shapes
:���������:5
1
/
_output_shapes
:���������:51
/
_output_shapes
:���������
�4
�
#__inference_internal_grad_fn_219836
result_grads_0
result_grads_1
result_grads_2
result_grads_30
,mul_distance_based_1_minkowski_1_broadcastto(
$mul_distance_based_1_minkowski_1_sub(
$pow_distance_based_1_minkowski_1_abs/
+div_no_nan_distance_based_1_minkowski_1_pow/
+sub_distance_based_1_minkowski_1_expanddims,
(pow_1_distance_based_1_minkowski_1_pow_11
-div_no_nan_1_distance_based_1_minkowski_1_sum,
(mul_6_distance_based_1_minkowski_1_mul_1
identity

identity_1

identity_2�
mulMul,mul_distance_based_1_minkowski_1_broadcastto$mul_distance_based_1_minkowski_1_sub*
T0*/
_output_shapes
:���������J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
powPow$pow_distance_based_1_minkowski_1_abspow/y:output:0*
T0*/
_output_shapes
:����������

div_no_nanDivNoNan+div_no_nan_distance_based_1_minkowski_1_powpow:z:0*
T0*/
_output_shapes
:���������_
mul_1Mulmul:z:0div_no_nan:z:0*
T0*/
_output_shapes
:���������J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
subSub+sub_distance_based_1_minkowski_1_expanddimssub/y:output:0*
T0*/
_output_shapes
:���������y
pow_1Pow(pow_1_distance_based_1_minkowski_1_pow_1sub:z:0*
T0*/
_output_shapes
:���������J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3a
addAddV2	pow_1:z:0add/y:output:0*
T0*/
_output_shapes
:���������`
truedivRealDiv	mul_1:z:0add:z:0*
T0*/
_output_shapes
:���������c
mul_2Mulresult_grads_0truediv:z:0*
T0*/
_output_shapes
:���������L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sub_1Sub+sub_distance_based_1_minkowski_1_expanddimssub_1/y:output:0*
T0*/
_output_shapes
:���������{
pow_2Pow(pow_1_distance_based_1_minkowski_1_pow_1	sub_1:z:0*
T0*/
_output_shapes
:���������~
mul_3Mul+sub_distance_based_1_minkowski_1_expanddims	pow_2:z:0*
T0*/
_output_shapes
:���������L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3e
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*/
_output_shapes
:����������
	truediv_1RealDiv+div_no_nan_distance_based_1_minkowski_1_pow	add_1:z:0*
T0*/
_output_shapes
:���������e
mul_4Mulresult_grads_0truediv_1:z:0*
T0*/
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
	truediv_2RealDivtruediv_2/x:output:0+sub_distance_based_1_minkowski_1_expanddims*
T0*/
_output_shapes
:����������
div_no_nan_1DivNoNan(pow_1_distance_based_1_minkowski_1_pow_1-div_no_nan_1_distance_based_1_minkowski_1_sum*
T0*/
_output_shapes
:���������g
mul_5Multruediv_2:z:0div_no_nan_1:z:0*
T0*/
_output_shapes
:���������L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
add_2AddV2$pow_distance_based_1_minkowski_1_absadd_2/y:output:0*
T0*/
_output_shapes
:���������O
LogLog	add_2:z:0*
T0*/
_output_shapes
:���������y
mul_6Mul(mul_6_distance_based_1_minkowski_1_mul_1Log:y:0*
T0*/
_output_shapes
:���������`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
SumSum	mul_6:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(_
mul_7Mul	mul_5:z:0Sum:output:0*
T0*/
_output_shapes
:���������L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
pow_3Pow+sub_distance_based_1_minkowski_1_expanddimspow_3/y:output:0*
T0*/
_output_shapes
:���������P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
	truediv_3RealDivtruediv_3/x:output:0	pow_3:z:0*
T0*/
_output_shapes
:���������
mul_8Multruediv_3:z:0(pow_1_distance_based_1_minkowski_1_pow_1*
T0*/
_output_shapes
:���������L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
add_3AddV2-div_no_nan_1_distance_based_1_minkowski_1_sumadd_3/y:output:0*
T0*/
_output_shapes
:���������Q
Log_1Log	add_3:z:0*
T0*/
_output_shapes
:���������\
mul_9Mul	mul_8:z:0	Log_1:y:0*
T0*/
_output_shapes
:���������\
sub_2Sub	mul_7:z:0	mul_9:z:0*
T0*/
_output_shapes
:���������b
mul_10Mulresult_grads_0	sub_2:z:0*
T0*/
_output_shapes
:���������t
SqueezeSqueeze
mul_10:z:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:���������[

Identity_1Identity	mul_4:z:0*
T0*/
_output_shapes
:���������^

Identity_2IdentitySqueeze:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:_ [
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_1:_[
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:���������
(
_user_specified_nameresult_grads_3:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:5	1
/
_output_shapes
:���������:5
1
/
_output_shapes
:���������:51
/
_output_shapes
:���������
�4
�
#__inference_internal_grad_fn_219958
result_grads_0
result_grads_1
result_grads_2
result_grads_30
,mul_distance_based_1_minkowski_1_broadcastto(
$mul_distance_based_1_minkowski_1_sub(
$pow_distance_based_1_minkowski_1_abs/
+div_no_nan_distance_based_1_minkowski_1_pow/
+sub_distance_based_1_minkowski_1_expanddims,
(pow_1_distance_based_1_minkowski_1_pow_11
-div_no_nan_1_distance_based_1_minkowski_1_sum,
(mul_6_distance_based_1_minkowski_1_mul_1
identity

identity_1

identity_2�
mulMul,mul_distance_based_1_minkowski_1_broadcastto$mul_distance_based_1_minkowski_1_sub*
T0*/
_output_shapes
:���������J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
powPow$pow_distance_based_1_minkowski_1_abspow/y:output:0*
T0*/
_output_shapes
:����������

div_no_nanDivNoNan+div_no_nan_distance_based_1_minkowski_1_powpow:z:0*
T0*/
_output_shapes
:���������_
mul_1Mulmul:z:0div_no_nan:z:0*
T0*/
_output_shapes
:���������J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
subSub+sub_distance_based_1_minkowski_1_expanddimssub/y:output:0*
T0*/
_output_shapes
:���������y
pow_1Pow(pow_1_distance_based_1_minkowski_1_pow_1sub:z:0*
T0*/
_output_shapes
:���������J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3a
addAddV2	pow_1:z:0add/y:output:0*
T0*/
_output_shapes
:���������`
truedivRealDiv	mul_1:z:0add:z:0*
T0*/
_output_shapes
:���������c
mul_2Mulresult_grads_0truediv:z:0*
T0*/
_output_shapes
:���������L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sub_1Sub+sub_distance_based_1_minkowski_1_expanddimssub_1/y:output:0*
T0*/
_output_shapes
:���������{
pow_2Pow(pow_1_distance_based_1_minkowski_1_pow_1	sub_1:z:0*
T0*/
_output_shapes
:���������~
mul_3Mul+sub_distance_based_1_minkowski_1_expanddims	pow_2:z:0*
T0*/
_output_shapes
:���������L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3e
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*/
_output_shapes
:����������
	truediv_1RealDiv+div_no_nan_distance_based_1_minkowski_1_pow	add_1:z:0*
T0*/
_output_shapes
:���������e
mul_4Mulresult_grads_0truediv_1:z:0*
T0*/
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
	truediv_2RealDivtruediv_2/x:output:0+sub_distance_based_1_minkowski_1_expanddims*
T0*/
_output_shapes
:����������
div_no_nan_1DivNoNan(pow_1_distance_based_1_minkowski_1_pow_1-div_no_nan_1_distance_based_1_minkowski_1_sum*
T0*/
_output_shapes
:���������g
mul_5Multruediv_2:z:0div_no_nan_1:z:0*
T0*/
_output_shapes
:���������L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
add_2AddV2$pow_distance_based_1_minkowski_1_absadd_2/y:output:0*
T0*/
_output_shapes
:���������O
LogLog	add_2:z:0*
T0*/
_output_shapes
:���������y
mul_6Mul(mul_6_distance_based_1_minkowski_1_mul_1Log:y:0*
T0*/
_output_shapes
:���������`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
SumSum	mul_6:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(_
mul_7Mul	mul_5:z:0Sum:output:0*
T0*/
_output_shapes
:���������L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
pow_3Pow+sub_distance_based_1_minkowski_1_expanddimspow_3/y:output:0*
T0*/
_output_shapes
:���������P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
	truediv_3RealDivtruediv_3/x:output:0	pow_3:z:0*
T0*/
_output_shapes
:���������
mul_8Multruediv_3:z:0(pow_1_distance_based_1_minkowski_1_pow_1*
T0*/
_output_shapes
:���������L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
add_3AddV2-div_no_nan_1_distance_based_1_minkowski_1_sumadd_3/y:output:0*
T0*/
_output_shapes
:���������Q
Log_1Log	add_3:z:0*
T0*/
_output_shapes
:���������\
mul_9Mul	mul_8:z:0	Log_1:y:0*
T0*/
_output_shapes
:���������\
sub_2Sub	mul_7:z:0	mul_9:z:0*
T0*/
_output_shapes
:���������b
mul_10Mulresult_grads_0	sub_2:z:0*
T0*/
_output_shapes
:���������t
SqueezeSqueeze
mul_10:z:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:���������[

Identity_1Identity	mul_4:z:0*
T0*/
_output_shapes
:���������^

Identity_2IdentitySqueeze:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:_ [
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_1:_[
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:���������
(
_user_specified_nameresult_grads_3:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:5	1
/
_output_shapes
:���������:5
1
/
_output_shapes
:���������:51
/
_output_shapes
:���������
�F
�
"__inference__traced_restore_220239
file_prefix]
Jassignvariableop_behavior_model_1_rank_similarity_1_embedding_1_embeddings:	�,
"assignvariableop_1_minkowski_1_rho: b
Tassignvariableop_2_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_w:9
/assignvariableop_3_exponential_similarity_1_tau: ;
1assignvariableop_4_exponential_similarity_1_gamma: :
0assignvariableop_5_exponential_similarity_1_beta: &
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: #
assignvariableop_13_total: #
assignvariableop_14_count: g
Tassignvariableop_15_adam_behavior_model_1_rank_similarity_1_embedding_1_embeddings_m:	�g
Tassignvariableop_16_adam_behavior_model_1_rank_similarity_1_embedding_1_embeddings_v:	�
identity_18��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpJassignvariableop_behavior_model_1_rank_similarity_1_embedding_1_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp"assignvariableop_1_minkowski_1_rhoIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpTassignvariableop_2_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_wIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp/assignvariableop_3_exponential_similarity_1_tauIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp1assignvariableop_4_exponential_similarity_1_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp0assignvariableop_5_exponential_similarity_1_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpTassignvariableop_15_adam_behavior_model_1_rank_similarity_1_embedding_1_embeddings_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpTassignvariableop_16_adam_behavior_model_1_rank_similarity_1_embedding_1_embeddings_vIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_17Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_18IdentityIdentity_17:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_18Identity_18:output:0*7
_input_shapes&
$: : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�8
�
#__inference_internal_grad_fn_220019
result_grads_0
result_grads_1
result_grads_2
result_grads_3B
>mul_rank_similarity_1_distance_based_1_minkowski_1_broadcastto:
6mul_rank_similarity_1_distance_based_1_minkowski_1_sub:
6pow_rank_similarity_1_distance_based_1_minkowski_1_absA
=div_no_nan_rank_similarity_1_distance_based_1_minkowski_1_powA
=sub_rank_similarity_1_distance_based_1_minkowski_1_expanddims>
:pow_1_rank_similarity_1_distance_based_1_minkowski_1_pow_1C
?div_no_nan_1_rank_similarity_1_distance_based_1_minkowski_1_sum>
:mul_6_rank_similarity_1_distance_based_1_minkowski_1_mul_1
identity

identity_1

identity_2�
mulMul>mul_rank_similarity_1_distance_based_1_minkowski_1_broadcastto6mul_rank_similarity_1_distance_based_1_minkowski_1_sub*
T0*/
_output_shapes
:���������J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
powPow6pow_rank_similarity_1_distance_based_1_minkowski_1_abspow/y:output:0*
T0*/
_output_shapes
:����������

div_no_nanDivNoNan=div_no_nan_rank_similarity_1_distance_based_1_minkowski_1_powpow:z:0*
T0*/
_output_shapes
:���������_
mul_1Mulmul:z:0div_no_nan:z:0*
T0*/
_output_shapes
:���������J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
subSub=sub_rank_similarity_1_distance_based_1_minkowski_1_expanddimssub/y:output:0*
T0*/
_output_shapes
:����������
pow_1Pow:pow_1_rank_similarity_1_distance_based_1_minkowski_1_pow_1sub:z:0*
T0*/
_output_shapes
:���������J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3a
addAddV2	pow_1:z:0add/y:output:0*
T0*/
_output_shapes
:���������`
truedivRealDiv	mul_1:z:0add:z:0*
T0*/
_output_shapes
:���������c
mul_2Mulresult_grads_0truediv:z:0*
T0*/
_output_shapes
:���������L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sub_1Sub=sub_rank_similarity_1_distance_based_1_minkowski_1_expanddimssub_1/y:output:0*
T0*/
_output_shapes
:����������
pow_2Pow:pow_1_rank_similarity_1_distance_based_1_minkowski_1_pow_1	sub_1:z:0*
T0*/
_output_shapes
:����������
mul_3Mul=sub_rank_similarity_1_distance_based_1_minkowski_1_expanddims	pow_2:z:0*
T0*/
_output_shapes
:���������L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3e
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*/
_output_shapes
:����������
	truediv_1RealDiv=div_no_nan_rank_similarity_1_distance_based_1_minkowski_1_pow	add_1:z:0*
T0*/
_output_shapes
:���������e
mul_4Mulresult_grads_0truediv_1:z:0*
T0*/
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
	truediv_2RealDivtruediv_2/x:output:0=sub_rank_similarity_1_distance_based_1_minkowski_1_expanddims*
T0*/
_output_shapes
:����������
div_no_nan_1DivNoNan:pow_1_rank_similarity_1_distance_based_1_minkowski_1_pow_1?div_no_nan_1_rank_similarity_1_distance_based_1_minkowski_1_sum*
T0*/
_output_shapes
:���������g
mul_5Multruediv_2:z:0div_no_nan_1:z:0*
T0*/
_output_shapes
:���������L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
add_2AddV26pow_rank_similarity_1_distance_based_1_minkowski_1_absadd_2/y:output:0*
T0*/
_output_shapes
:���������O
LogLog	add_2:z:0*
T0*/
_output_shapes
:����������
mul_6Mul:mul_6_rank_similarity_1_distance_based_1_minkowski_1_mul_1Log:y:0*
T0*/
_output_shapes
:���������`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
SumSum	mul_6:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(_
mul_7Mul	mul_5:z:0Sum:output:0*
T0*/
_output_shapes
:���������L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
pow_3Pow=sub_rank_similarity_1_distance_based_1_minkowski_1_expanddimspow_3/y:output:0*
T0*/
_output_shapes
:���������P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
	truediv_3RealDivtruediv_3/x:output:0	pow_3:z:0*
T0*/
_output_shapes
:����������
mul_8Multruediv_3:z:0:pow_1_rank_similarity_1_distance_based_1_minkowski_1_pow_1*
T0*/
_output_shapes
:���������L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
add_3AddV2?div_no_nan_1_rank_similarity_1_distance_based_1_minkowski_1_sumadd_3/y:output:0*
T0*/
_output_shapes
:���������Q
Log_1Log	add_3:z:0*
T0*/
_output_shapes
:���������\
mul_9Mul	mul_8:z:0	Log_1:y:0*
T0*/
_output_shapes
:���������\
sub_2Sub	mul_7:z:0	mul_9:z:0*
T0*/
_output_shapes
:���������b
mul_10Mulresult_grads_0	sub_2:z:0*
T0*/
_output_shapes
:���������t
SqueezeSqueeze
mul_10:z:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:���������[

Identity_1Identity	mul_4:z:0*
T0*/
_output_shapes
:���������^

Identity_2IdentitySqueeze:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:_ [
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_1:_[
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:���������
(
_user_specified_nameresult_grads_3:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:5	1
/
_output_shapes
:���������:5
1
/
_output_shapes
:���������:51
/
_output_shapes
:���������
�4
�
#__inference_internal_grad_fn_219897
result_grads_0
result_grads_1
result_grads_2
result_grads_30
,mul_distance_based_1_minkowski_1_broadcastto(
$mul_distance_based_1_minkowski_1_sub(
$pow_distance_based_1_minkowski_1_abs/
+div_no_nan_distance_based_1_minkowski_1_pow/
+sub_distance_based_1_minkowski_1_expanddims,
(pow_1_distance_based_1_minkowski_1_pow_11
-div_no_nan_1_distance_based_1_minkowski_1_sum,
(mul_6_distance_based_1_minkowski_1_mul_1
identity

identity_1

identity_2�
mulMul,mul_distance_based_1_minkowski_1_broadcastto$mul_distance_based_1_minkowski_1_sub*
T0*/
_output_shapes
:���������J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
powPow$pow_distance_based_1_minkowski_1_abspow/y:output:0*
T0*/
_output_shapes
:����������

div_no_nanDivNoNan+div_no_nan_distance_based_1_minkowski_1_powpow:z:0*
T0*/
_output_shapes
:���������_
mul_1Mulmul:z:0div_no_nan:z:0*
T0*/
_output_shapes
:���������J
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
subSub+sub_distance_based_1_minkowski_1_expanddimssub/y:output:0*
T0*/
_output_shapes
:���������y
pow_1Pow(pow_1_distance_based_1_minkowski_1_pow_1sub:z:0*
T0*/
_output_shapes
:���������J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3a
addAddV2	pow_1:z:0add/y:output:0*
T0*/
_output_shapes
:���������`
truedivRealDiv	mul_1:z:0add:z:0*
T0*/
_output_shapes
:���������c
mul_2Mulresult_grads_0truediv:z:0*
T0*/
_output_shapes
:���������L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
sub_1Sub+sub_distance_based_1_minkowski_1_expanddimssub_1/y:output:0*
T0*/
_output_shapes
:���������{
pow_2Pow(pow_1_distance_based_1_minkowski_1_pow_1	sub_1:z:0*
T0*/
_output_shapes
:���������~
mul_3Mul+sub_distance_based_1_minkowski_1_expanddims	pow_2:z:0*
T0*/
_output_shapes
:���������L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3e
add_1AddV2	mul_3:z:0add_1/y:output:0*
T0*/
_output_shapes
:����������
	truediv_1RealDiv+div_no_nan_distance_based_1_minkowski_1_pow	add_1:z:0*
T0*/
_output_shapes
:���������e
mul_4Mulresult_grads_0truediv_1:z:0*
T0*/
_output_shapes
:���������P
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
	truediv_2RealDivtruediv_2/x:output:0+sub_distance_based_1_minkowski_1_expanddims*
T0*/
_output_shapes
:����������
div_no_nan_1DivNoNan(pow_1_distance_based_1_minkowski_1_pow_1-div_no_nan_1_distance_based_1_minkowski_1_sum*
T0*/
_output_shapes
:���������g
mul_5Multruediv_2:z:0div_no_nan_1:z:0*
T0*/
_output_shapes
:���������L
add_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
add_2AddV2$pow_distance_based_1_minkowski_1_absadd_2/y:output:0*
T0*/
_output_shapes
:���������O
LogLog	add_2:z:0*
T0*/
_output_shapes
:���������y
mul_6Mul(mul_6_distance_based_1_minkowski_1_mul_1Log:y:0*
T0*/
_output_shapes
:���������`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
SumSum	mul_6:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(_
mul_7Mul	mul_5:z:0Sum:output:0*
T0*/
_output_shapes
:���������L
pow_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
pow_3Pow+sub_distance_based_1_minkowski_1_expanddimspow_3/y:output:0*
T0*/
_output_shapes
:���������P
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?o
	truediv_3RealDivtruediv_3/x:output:0	pow_3:z:0*
T0*/
_output_shapes
:���������
mul_8Multruediv_3:z:0(pow_1_distance_based_1_minkowski_1_pow_1*
T0*/
_output_shapes
:���������L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
add_3AddV2-div_no_nan_1_distance_based_1_minkowski_1_sumadd_3/y:output:0*
T0*/
_output_shapes
:���������Q
Log_1Log	add_3:z:0*
T0*/
_output_shapes
:���������\
mul_9Mul	mul_8:z:0	Log_1:y:0*
T0*/
_output_shapes
:���������\
sub_2Sub	mul_7:z:0	mul_9:z:0*
T0*/
_output_shapes
:���������b
mul_10Mulresult_grads_0	sub_2:z:0*
T0*/
_output_shapes
:���������t
SqueezeSqueeze
mul_10:z:0*
T0*+
_output_shapes
:���������*
squeeze_dims

���������Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:���������[

Identity_1Identity	mul_4:z:0*
T0*/
_output_shapes
:���������^

Identity_2IdentitySqueeze:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*�
_input_shapes�
�:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:���������:_ [
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_1:_[
/
_output_shapes
:���������
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:���������
(
_user_specified_nameresult_grads_3:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:51
/
_output_shapes
:���������:5	1
/
_output_shapes
:���������:5
1
/
_output_shapes
:���������:51
/
_output_shapes
:���������
�+
�
__inference__traced_save_220178
file_prefixX
Tsavev2_behavior_model_1_rank_similarity_1_embedding_1_embeddings_read_readvariableop.
*savev2_minkowski_1_rho_read_readvariableop`
\savev2_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_w_read_readvariableop;
7savev2_exponential_similarity_1_tau_read_readvariableop=
9savev2_exponential_similarity_1_gamma_read_readvariableop<
8savev2_exponential_similarity_1_beta_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop_
[savev2_adam_behavior_model_1_rank_similarity_1_embedding_1_embeddings_m_read_readvariableop_
[savev2_adam_behavior_model_1_rank_similarity_1_embedding_1_embeddings_v_read_readvariableop
savev2_const_6

identity_1��MergeV2Checkpointsw
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Tsavev2_behavior_model_1_rank_similarity_1_embedding_1_embeddings_read_readvariableop*savev2_minkowski_1_rho_read_readvariableop\savev2_behavior_model_1_rank_similarity_1_distance_based_1_minkowski_1_w_read_readvariableop7savev2_exponential_similarity_1_tau_read_readvariableop9savev2_exponential_similarity_1_gamma_read_readvariableop8savev2_exponential_similarity_1_beta_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop[savev2_adam_behavior_model_1_rank_similarity_1_embedding_1_embeddings_m_read_readvariableop[savev2_adam_behavior_model_1_rank_similarity_1_embedding_1_embeddings_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 * 
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*X
_input_shapesG
E: :	�: :: : : : : : : : : : : : :	�:	�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�:

_output_shapes
: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�:%!

_output_shapes
:	�:

_output_shapes
: 
�
�
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219101
rank2_stimulus_set	
rank_similarity_1_219075
rank_similarity_1_219077+
rank_similarity_1_219079:	�
rank_similarity_1_219081"
rank_similarity_1_219083: &
rank_similarity_1_219085:"
rank_similarity_1_219087: "
rank_similarity_1_219089: "
rank_similarity_1_219091: 
rank_similarity_1_219093
rank_similarity_1_219095
rank_similarity_1_219097
identity��)rank_similarity_1/StatefulPartitionedCall�
)rank_similarity_1/StatefulPartitionedCallStatefulPartitionedCallrank2_stimulus_setrank_similarity_1_219075rank_similarity_1_219077rank_similarity_1_219079rank_similarity_1_219081rank_similarity_1_219083rank_similarity_1_219085rank_similarity_1_219087rank_similarity_1_219089rank_similarity_1_219091rank_similarity_1_219093rank_similarity_1_219095rank_similarity_1_219097*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������8*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218746�
IdentityIdentity2rank_similarity_1/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������8r
NoOpNoOp*^rank_similarity_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 2V
)rank_similarity_1/StatefulPartitionedCall)rank_similarity_1/StatefulPartitionedCall:` \
+
_output_shapes
:���������	
-
_user_specified_name8rank2/stimulus_set: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: 
��
�

L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219322
inputs_8rank2_stimulus_set	&
"rank_similarity_1_gatherv2_indices#
rank_similarity_1_gatherv2_axisH
5rank_similarity_1_embedding_1_embedding_lookup_219233:	�%
!rank_similarity_1_packed_values_1P
Frank_similarity_1_distance_based_1_minkowski_1_readvariableop_resource: `
Rrank_similarity_1_distance_based_1_minkowski_1_broadcastto_readvariableop_resource:a
Wrank_similarity_1_distance_based_1_exponential_similarity_1_neg_readvariableop_resource: a
Wrank_similarity_1_distance_based_1_exponential_similarity_1_pow_readvariableop_resource: a
Wrank_similarity_1_distance_based_1_exponential_similarity_1_add_readvariableop_resource: (
$rank_similarity_1_gatherv2_1_indices
rank_similarity_1_mul_x
rank_similarity_1_truediv_y
identity��Nrank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOp�Nrank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOp�Nrank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOp�Irank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp�=rank_similarity_1/distance_based_1/minkowski_1/ReadVariableOp�.rank_similarity_1/embedding_1/embedding_lookup�
rank_similarity_1/GatherV2GatherV2inputs_8rank2_stimulus_set"rank_similarity_1_gatherv2_indicesrank_similarity_1_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0	*+
_output_shapes
:���������^
rank_similarity_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
rank_similarity_1/NotEqualNotEqual#rank_similarity_1/GatherV2:output:0%rank_similarity_1/NotEqual/y:output:0*
T0	*+
_output_shapes
:����������
.rank_similarity_1/embedding_1/embedding_lookupResourceGather5rank_similarity_1_embedding_1_embedding_lookup_219233inputs_8rank2_stimulus_set*
Tindices0	*H
_class>
<:loc:@rank_similarity_1/embedding_1/embedding_lookup/219233*/
_output_shapes
:���������	*
dtype0�
7rank_similarity_1/embedding_1/embedding_lookup/IdentityIdentity7rank_similarity_1/embedding_1/embedding_lookup:output:0*
T0*H
_class>
<:loc:@rank_similarity_1/embedding_1/embedding_lookup/219233*/
_output_shapes
:���������	�
9rank_similarity_1/embedding_1/embedding_lookup/Identity_1Identity@rank_similarity_1/embedding_1/embedding_lookup/Identity:output:0*
T0*/
_output_shapes
:���������	j
(rank_similarity_1/embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
&rank_similarity_1/embedding_1/NotEqualNotEqualinputs_8rank2_stimulus_set1rank_similarity_1/embedding_1/NotEqual/y:output:0*
T0	*+
_output_shapes
:���������	\
rank_similarity_1/packed/0Const*
_output_shapes
: *
dtype0*
value	B :�
rank_similarity_1/packedPack#rank_similarity_1/packed/0:output:0!rank_similarity_1_packed_values_1*
N*
T0*
_output_shapes
:c
!rank_similarity_1/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
rank_similarity_1/splitSplitVBrank_similarity_1/embedding_1/embedding_lookup/Identity_1:output:0!rank_similarity_1/packed:output:0*rank_similarity_1/split/split_dim:output:0*
T0*

Tlen0*J
_output_shapes8
6:���������:���������*
	num_split�
2rank_similarity_1/distance_based_1/minkowski_1/subSub rank_similarity_1/split:output:0 rank_similarity_1/split:output:1*
T0*/
_output_shapes
:����������
4rank_similarity_1/distance_based_1/minkowski_1/ShapeShape6rank_similarity_1/distance_based_1/minkowski_1/sub:z:0*
T0*
_output_shapes
:�
Brank_similarity_1/distance_based_1/minkowski_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Drank_similarity_1/distance_based_1/minkowski_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
����������
Drank_similarity_1/distance_based_1/minkowski_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
<rank_similarity_1/distance_based_1/minkowski_1/strided_sliceStridedSlice=rank_similarity_1/distance_based_1/minkowski_1/Shape:output:0Krank_similarity_1/distance_based_1/minkowski_1/strided_slice/stack:output:0Mrank_similarity_1/distance_based_1/minkowski_1/strided_slice/stack_1:output:0Mrank_similarity_1/distance_based_1/minkowski_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:~
9rank_similarity_1/distance_based_1/minkowski_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
3rank_similarity_1/distance_based_1/minkowski_1/onesFillErank_similarity_1/distance_based_1/minkowski_1/strided_slice:output:0Brank_similarity_1/distance_based_1/minkowski_1/ones/Const:output:0*
T0*+
_output_shapes
:����������
=rank_similarity_1/distance_based_1/minkowski_1/ReadVariableOpReadVariableOpFrank_similarity_1_distance_based_1_minkowski_1_readvariableop_resource*
_output_shapes
: *
dtype0�
2rank_similarity_1/distance_based_1/minkowski_1/mulMulErank_similarity_1/distance_based_1/minkowski_1/ReadVariableOp:value:0<rank_similarity_1/distance_based_1/minkowski_1/ones:output:0*
T0*+
_output_shapes
:����������
Irank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOpReadVariableOpRrank_similarity_1_distance_based_1_minkowski_1_broadcastto_readvariableop_resource*
_output_shapes
:*
dtype0�
:rank_similarity_1/distance_based_1/minkowski_1/BroadcastToBroadcastToQrank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp:value:0=rank_similarity_1/distance_based_1/minkowski_1/Shape:output:0*
T0*/
_output_shapes
:����������
2rank_similarity_1/distance_based_1/minkowski_1/AbsAbs6rank_similarity_1/distance_based_1/minkowski_1/sub:z:0*
T0*/
_output_shapes
:����������
=rank_similarity_1/distance_based_1/minkowski_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
9rank_similarity_1/distance_based_1/minkowski_1/ExpandDims
ExpandDims6rank_similarity_1/distance_based_1/minkowski_1/mul:z:0Frank_similarity_1/distance_based_1/minkowski_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:����������
2rank_similarity_1/distance_based_1/minkowski_1/PowPow6rank_similarity_1/distance_based_1/minkowski_1/Abs:y:0Brank_similarity_1/distance_based_1/minkowski_1/ExpandDims:output:0*
T0*/
_output_shapes
:����������
4rank_similarity_1/distance_based_1/minkowski_1/Mul_1Mul6rank_similarity_1/distance_based_1/minkowski_1/Pow:z:0Crank_similarity_1/distance_based_1/minkowski_1/BroadcastTo:output:0*
T0*/
_output_shapes
:����������
Drank_similarity_1/distance_based_1/minkowski_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
2rank_similarity_1/distance_based_1/minkowski_1/SumSum8rank_similarity_1/distance_based_1/minkowski_1/Mul_1:z:0Mrank_similarity_1/distance_based_1/minkowski_1/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:���������*
	keep_dims(}
8rank_similarity_1/distance_based_1/minkowski_1/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
6rank_similarity_1/distance_based_1/minkowski_1/truedivRealDivArank_similarity_1/distance_based_1/minkowski_1/truediv/x:output:0Brank_similarity_1/distance_based_1/minkowski_1/ExpandDims:output:0*
T0*/
_output_shapes
:����������
4rank_similarity_1/distance_based_1/minkowski_1/Pow_1Pow;rank_similarity_1/distance_based_1/minkowski_1/Sum:output:0:rank_similarity_1/distance_based_1/minkowski_1/truediv:z:0*
T0*/
_output_shapes
:����������
7rank_similarity_1/distance_based_1/minkowski_1/IdentityIdentity8rank_similarity_1/distance_based_1/minkowski_1/Pow_1:z:0*
T0*/
_output_shapes
:����������
8rank_similarity_1/distance_based_1/minkowski_1/IdentityN	IdentityN8rank_similarity_1/distance_based_1/minkowski_1/Pow_1:z:06rank_similarity_1/distance_based_1/minkowski_1/sub:z:0Crank_similarity_1/distance_based_1/minkowski_1/BroadcastTo:output:06rank_similarity_1/distance_based_1/minkowski_1/mul:z:0*
T
2*,
_gradient_op_typeCustomGradient-219260*|
_output_shapesj
h:���������:���������:���������:����������
6rank_similarity_1/distance_based_1/minkowski_1/SqueezeSqueezeArank_similarity_1/distance_based_1/minkowski_1/IdentityN:output:0*
T0*+
_output_shapes
:���������*
squeeze_dims

����������
Nrank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOpReadVariableOpWrank_similarity_1_distance_based_1_exponential_similarity_1_neg_readvariableop_resource*
_output_shapes
: *
dtype0�
?rank_similarity_1/distance_based_1/exponential_similarity_1/NegNegVrank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOp:value:0*
T0*
_output_shapes
: �
Nrank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOpReadVariableOpWrank_similarity_1_distance_based_1_exponential_similarity_1_pow_readvariableop_resource*
_output_shapes
: *
dtype0�
?rank_similarity_1/distance_based_1/exponential_similarity_1/PowPow?rank_similarity_1/distance_based_1/minkowski_1/Squeeze:output:0Vrank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
?rank_similarity_1/distance_based_1/exponential_similarity_1/mulMulCrank_similarity_1/distance_based_1/exponential_similarity_1/Neg:y:0Crank_similarity_1/distance_based_1/exponential_similarity_1/Pow:z:0*
T0*+
_output_shapes
:����������
?rank_similarity_1/distance_based_1/exponential_similarity_1/ExpExpCrank_similarity_1/distance_based_1/exponential_similarity_1/mul:z:0*
T0*+
_output_shapes
:����������
Nrank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOpReadVariableOpWrank_similarity_1_distance_based_1_exponential_similarity_1_add_readvariableop_resource*
_output_shapes
: *
dtype0�
?rank_similarity_1/distance_based_1/exponential_similarity_1/addAddV2Crank_similarity_1/distance_based_1/exponential_similarity_1/Exp:y:0Vrank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:����������
rank_similarity_1/CastCastrank_similarity_1/NotEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:����������
.rank_similarity_1/rank_sim_zero_out_nonpresentMulCrank_similarity_1/distance_based_1/exponential_similarity_1/add:z:0rank_similarity_1/Cast:y:0*
T0*+
_output_shapes
:����������
rank_similarity_1/GatherV2_1GatherV22rank_similarity_1/rank_sim_zero_out_nonpresent:z:0$rank_similarity_1_gatherv2_1_indicesrank_similarity_1_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*/
_output_shapes
:���������8b
 rank_similarity_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
rank_similarity_1/ExpandDims
ExpandDimsrank_similarity_1/Cast:y:0)rank_similarity_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������Y
rank_similarity_1/ConstConst*
_output_shapes
: *
dtype0*
value	B : �
rank_similarity_1/GatherV2_2GatherV2%rank_similarity_1/ExpandDims:output:0 rank_similarity_1/Const:output:0rank_similarity_1_gatherv2_axis*
Taxis0*
Tindices0*
Tparams0*+
_output_shapes
:���������_
rank_similarity_1/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B :�
rank_similarity_1/CumsumCumsum%rank_similarity_1/GatherV2_1:output:0&rank_similarity_1/Cumsum/axis:output:0*
T0*/
_output_shapes
:���������8*
reverse(`
rank_similarity_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
rank_similarity_1/MaximumMaximum%rank_similarity_1/GatherV2_1:output:0$rank_similarity_1/Maximum/y:output:0*
T0*/
_output_shapes
:���������8b
rank_similarity_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
rank_similarity_1/Maximum_1Maximumrank_similarity_1/Cumsum:out:0&rank_similarity_1/Maximum_1/y:output:0*
T0*/
_output_shapes
:���������8u
rank_similarity_1/LogLogrank_similarity_1/Maximum:z:0*
T0*/
_output_shapes
:���������8y
rank_similarity_1/Log_1Logrank_similarity_1/Maximum_1:z:0*
T0*/
_output_shapes
:���������8�
rank_similarity_1/subSubrank_similarity_1/Log:y:0rank_similarity_1/Log_1:y:0*
T0*/
_output_shapes
:���������8�
rank_similarity_1/mulMulrank_similarity_1_mul_xrank_similarity_1/sub:z:0*
T0*/
_output_shapes
:���������8i
'rank_similarity_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
rank_similarity_1/SumSumrank_similarity_1/mul:z:00rank_similarity_1/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:���������8r
rank_similarity_1/ExpExprank_similarity_1/Sum:output:0*
T0*+
_output_shapes
:���������8�
rank_similarity_1/mul_1Mul%rank_similarity_1/GatherV2_2:output:0rank_similarity_1/Exp:y:0*
T0*+
_output_shapes
:���������8k
)rank_similarity_1/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
rank_similarity_1/Sum_1Sumrank_similarity_1/mul_1:z:02rank_similarity_1/Sum_1/reduction_indices:output:0*
T0*+
_output_shapes
:���������*
	keep_dims(^
rank_similarity_1/Equal/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
rank_similarity_1/EqualEqual rank_similarity_1/Sum_1:output:0"rank_similarity_1/Equal/y:output:0*
T0*+
_output_shapes
:����������
rank_similarity_1/Cast_1Castrank_similarity_1/Equal:z:0*

DstT0*

SrcT0
*+
_output_shapes
:����������
rank_similarity_1/truedivRealDivrank_similarity_1/Cast_1:y:0rank_similarity_1_truediv_y*
T0*+
_output_shapes
:����������
rank_similarity_1/addAddV2rank_similarity_1/mul_1:z:0rank_similarity_1/truediv:z:0*
T0*+
_output_shapes
:���������8�
rank_similarity_1/add_1AddV2 rank_similarity_1/Sum_1:output:0rank_similarity_1/Cast_1:y:0*
T0*+
_output_shapes
:����������
rank_similarity_1/truediv_1RealDivrank_similarity_1/add:z:0rank_similarity_1/add_1:z:0*
T0*+
_output_shapes
:���������8r
IdentityIdentityrank_similarity_1/truediv_1:z:0^NoOp*
T0*+
_output_shapes
:���������8�
NoOpNoOpO^rank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOpO^rank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOpO^rank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOpJ^rank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp>^rank_similarity_1/distance_based_1/minkowski_1/ReadVariableOp/^rank_similarity_1/embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 2�
Nrank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOpNrank_similarity_1/distance_based_1/exponential_similarity_1/Neg/ReadVariableOp2�
Nrank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOpNrank_similarity_1/distance_based_1/exponential_similarity_1/Pow/ReadVariableOp2�
Nrank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOpNrank_similarity_1/distance_based_1/exponential_similarity_1/add/ReadVariableOp2�
Irank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOpIrank_similarity_1/distance_based_1/minkowski_1/BroadcastTo/ReadVariableOp2~
=rank_similarity_1/distance_based_1/minkowski_1/ReadVariableOp=rank_similarity_1/distance_based_1/minkowski_1/ReadVariableOp2`
.rank_similarity_1/embedding_1/embedding_lookup.rank_similarity_1/embedding_1/embedding_lookup:g c
+
_output_shapes
:���������	
4
_user_specified_nameinputs/8rank2/stimulus_set: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: 
�
�
1__inference_behavior_model_1_layer_call_fn_218800
rank2_stimulus_set	
unknown
	unknown_0
	unknown_1:	�
	unknown_2
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallrank2_stimulus_setunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������8*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_218773s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������8`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:���������	
-
_user_specified_name8rank2/stimulus_set: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: 
�
�
2__inference_rank_similarity_1_layer_call_fn_219448
inputs_8rank2_stimulus_set	
unknown
	unknown_0
	unknown_1:	�
	unknown_2
	unknown_3: 
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8
	unknown_9

unknown_10
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_8rank2_stimulus_setunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������8*(
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218746s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������8`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:���������	:: : : : : : : : :8:: 22
StatefulPartitionedCallStatefulPartitionedCall:g c
+
_output_shapes
:���������	
4
_user_specified_nameinputs/8rank2/stimulus_set: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$
 

_output_shapes

:8:,(
&
_output_shapes
::

_output_shapes
: <
#__inference_internal_grad_fn_219775CustomGradient-219512<
#__inference_internal_grad_fn_219836CustomGradient-219609<
#__inference_internal_grad_fn_219897CustomGradient-218867<
#__inference_internal_grad_fn_219958CustomGradient-218684<
#__inference_internal_grad_fn_220019CustomGradient-219357<
#__inference_internal_grad_fn_220080CustomGradient-219260<
#__inference_internal_grad_fn_220141CustomGradient-218580"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
W
8rank2/stimulus_set@
%serving_default_8rank2_stimulus_set:0	���������	@
output_14
StatefulPartitionedCall:0���������8tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
behavior
		optimizer


signatures"
_tf_keras_model
J
0
1
2
3
4
5"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_1
trace_2
trace_32�
1__inference_behavior_model_1_layer_call_fn_218800
1__inference_behavior_model_1_layer_call_fn_219196
1__inference_behavior_model_1_layer_call_fn_219225
1__inference_behavior_model_1_layer_call_fn_219072�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1ztrace_2ztrace_3
�
trace_0
trace_1
trace_2
trace_32�
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219322
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219419
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219101
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219130�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 ztrace_0ztrace_1ztrace_2ztrace_3
�
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11B�
!__inference__wrapped_model_2186428rank2/stimulus_set"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_0z	capture_1z 	capture_3z!	capture_9z"
capture_10z#
capture_11
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*percept

+kernel
,percept_adapter
-kernel_adapter
.
_z_q_shape
/
_z_r_shape"
_tf_keras_layer
g
0iter

1beta_1

2beta_2
	3decay
4learning_ratem�v�"
	optimizer
,
5serving_default"
signature_map
L:J	�29behavior_model_1/rank_similarity_1/embedding_1/embeddings
: 2minkowski_1/rho
M:K2Abehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/w
$:" 2exponential_similarity_1/tau
&:$ 2exponential_similarity_1/gamma
%:# 2exponential_similarity_1/beta
C
0
1
2
3
4"
trackable_list_wrapper
'
0"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11B�
1__inference_behavior_model_1_layer_call_fn_2188008rank2/stimulus_set"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_0z	capture_1z 	capture_3z!	capture_9z"
capture_10z#
capture_11
�
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11B�
1__inference_behavior_model_1_layer_call_fn_219196inputs/8rank2/stimulus_set"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_0z	capture_1z 	capture_3z!	capture_9z"
capture_10z#
capture_11
�
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11B�
1__inference_behavior_model_1_layer_call_fn_219225inputs/8rank2/stimulus_set"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_0z	capture_1z 	capture_3z!	capture_9z"
capture_10z#
capture_11
�
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11B�
1__inference_behavior_model_1_layer_call_fn_2190728rank2/stimulus_set"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_0z	capture_1z 	capture_3z!	capture_9z"
capture_10z#
capture_11
�
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11B�
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219322inputs/8rank2/stimulus_set"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_0z	capture_1z 	capture_3z!	capture_9z"
capture_10z#
capture_11
�
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11B�
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219419inputs/8rank2/stimulus_set"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_0z	capture_1z 	capture_3z!	capture_9z"
capture_10z#
capture_11
�
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11B�
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_2191018rank2/stimulus_set"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_0z	capture_1z 	capture_3z!	capture_9z"
capture_10z#
capture_11
�
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11B�
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_2191308rank2/stimulus_set"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z	capture_0z	capture_1z 	capture_3z!	capture_9z"
capture_10z#
capture_11
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
J
0
1
2
3
4
5"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
=trace_0
>trace_12�
2__inference_rank_similarity_1_layer_call_fn_219448
2__inference_rank_similarity_1_layer_call_fn_219477�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z=trace_0z>trace_1
�
?trace_0
@trace_12�
M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219574
M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219671�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z?trace_0z@trace_1
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
Mdistance
N
similarity"
_tf_keras_layer
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses
U	_all_keys
V_input_keys
Wgating_keys"
_tf_keras_layer
�
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^	_all_keys
__input_keys
`gating_keys"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11B�
$__inference_signature_wrapper_2191678rank2/stimulus_set"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_0z	capture_1z 	capture_3z!	capture_9z"
capture_10z#
capture_11
N
a	variables
b	keras_api
	ctotal
	dcount"
_tf_keras_metric
^
e	variables
f	keras_api
	gtotal
	hcount
i
_fn_kwargs"
_tf_keras_metric
C
0
1
2
3
4"
trackable_list_wrapper
<
*0
+1
,2
-3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11B�
2__inference_rank_similarity_1_layer_call_fn_219448inputs/8rank2/stimulus_set"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_0z	capture_1z 	capture_3z!	capture_9z"
capture_10z#
capture_11
�
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11B�
2__inference_rank_similarity_1_layer_call_fn_219477inputs/8rank2/stimulus_set"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_0z	capture_1z 	capture_3z!	capture_9z"
capture_10z#
capture_11
�
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11B�
M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219574inputs/8rank2/stimulus_set"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_0z	capture_1z 	capture_3z!	capture_9z"
capture_10z#
capture_11
�
	capture_0
	capture_1
 	capture_3
!	capture_9
"
capture_10
#
capture_11B�
M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219671inputs/8rank2/stimulus_set"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z	capture_0z	capture_1z 	capture_3z!	capture_9z"
capture_10z#
capture_11
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses
rho
w"
_tf_keras_layer
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
tau
	gamma
beta"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
-
a	variables"
_generic_user_object
:  (2total
:  (2count
.
g0
h1"
trackable_list_wrapper
-
e	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Q:O	�2@Adam/behavior_model_1/rank_similarity_1/embedding_1/embeddings/m
Q:O	�2@Adam/behavior_model_1/rank_similarity_1/embedding_1/embeddings/v
}b{
*distance_based_1/minkowski_1/BroadcastTo:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219574
ubs
"distance_based_1/minkowski_1/sub:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219574
ubs
"distance_based_1/minkowski_1/Abs:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219574
ubs
"distance_based_1/minkowski_1/Pow:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219574
|bz
)distance_based_1/minkowski_1/ExpandDims:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219574
wbu
$distance_based_1/minkowski_1/Pow_1:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219574
ubs
"distance_based_1/minkowski_1/Sum:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219574
wbu
$distance_based_1/minkowski_1/Mul_1:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219574
}b{
*distance_based_1/minkowski_1/BroadcastTo:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219671
ubs
"distance_based_1/minkowski_1/sub:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219671
ubs
"distance_based_1/minkowski_1/Abs:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219671
ubs
"distance_based_1/minkowski_1/Pow:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219671
|bz
)distance_based_1/minkowski_1/ExpandDims:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219671
wbu
$distance_based_1/minkowski_1/Pow_1:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219671
ubs
"distance_based_1/minkowski_1/Sum:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219671
wbu
$distance_based_1/minkowski_1/Mul_1:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219671
}b{
*distance_based_1/minkowski_1/BroadcastTo:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218929
ubs
"distance_based_1/minkowski_1/sub:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218929
ubs
"distance_based_1/minkowski_1/Abs:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218929
ubs
"distance_based_1/minkowski_1/Pow:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218929
|bz
)distance_based_1/minkowski_1/ExpandDims:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218929
wbu
$distance_based_1/minkowski_1/Pow_1:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218929
ubs
"distance_based_1/minkowski_1/Sum:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218929
wbu
$distance_based_1/minkowski_1/Mul_1:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218929
}b{
*distance_based_1/minkowski_1/BroadcastTo:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218746
ubs
"distance_based_1/minkowski_1/sub:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218746
ubs
"distance_based_1/minkowski_1/Abs:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218746
ubs
"distance_based_1/minkowski_1/Pow:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218746
|bz
)distance_based_1/minkowski_1/ExpandDims:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218746
wbu
$distance_based_1/minkowski_1/Pow_1:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218746
ubs
"distance_based_1/minkowski_1/Sum:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218746
wbu
$distance_based_1/minkowski_1/Mul_1:0M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_218746
�b�
<rank_similarity_1/distance_based_1/minkowski_1/BroadcastTo:0L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219419
�b�
4rank_similarity_1/distance_based_1/minkowski_1/sub:0L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219419
�b�
4rank_similarity_1/distance_based_1/minkowski_1/Abs:0L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219419
�b�
4rank_similarity_1/distance_based_1/minkowski_1/Pow:0L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219419
�b�
;rank_similarity_1/distance_based_1/minkowski_1/ExpandDims:0L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219419
�b�
6rank_similarity_1/distance_based_1/minkowski_1/Pow_1:0L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219419
�b�
4rank_similarity_1/distance_based_1/minkowski_1/Sum:0L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219419
�b�
6rank_similarity_1/distance_based_1/minkowski_1/Mul_1:0L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219419
�b�
<rank_similarity_1/distance_based_1/minkowski_1/BroadcastTo:0L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219322
�b�
4rank_similarity_1/distance_based_1/minkowski_1/sub:0L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219322
�b�
4rank_similarity_1/distance_based_1/minkowski_1/Abs:0L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219322
�b�
4rank_similarity_1/distance_based_1/minkowski_1/Pow:0L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219322
�b�
;rank_similarity_1/distance_based_1/minkowski_1/ExpandDims:0L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219322
�b�
6rank_similarity_1/distance_based_1/minkowski_1/Pow_1:0L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219322
�b�
4rank_similarity_1/distance_based_1/minkowski_1/Sum:0L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219322
�b�
6rank_similarity_1/distance_based_1/minkowski_1/Mul_1:0L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219322
tbr
Mbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/BroadcastTo:0!__inference__wrapped_model_218642
lbj
Ebehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/sub:0!__inference__wrapped_model_218642
lbj
Ebehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Abs:0!__inference__wrapped_model_218642
lbj
Ebehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Pow:0!__inference__wrapped_model_218642
sbq
Lbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/ExpandDims:0!__inference__wrapped_model_218642
nbl
Gbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Pow_1:0!__inference__wrapped_model_218642
lbj
Ebehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Sum:0!__inference__wrapped_model_218642
nbl
Gbehavior_model_1/rank_similarity_1/distance_based_1/minkowski_1/Mul_1:0!__inference__wrapped_model_218642�
!__inference__wrapped_model_218642� !"#\�Y
R�O
M�J
H
8rank2/stimulus_set1�.
8rank2/stimulus_set���������		
� "7�4
2
output_1&�#
output_1���������8�
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219101� !"#l�i
R�O
M�J
H
8rank2/stimulus_set1�.
8rank2/stimulus_set���������		
�

trainingp ")�&
�
0���������8
� �
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219130� !"#l�i
R�O
M�J
H
8rank2/stimulus_set1�.
8rank2/stimulus_set���������		
�

trainingp")�&
�
0���������8
� �
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219322� !"#s�p
Y�V
T�Q
O
8rank2/stimulus_set8�5
inputs/8rank2/stimulus_set���������		
�

trainingp ")�&
�
0���������8
� �
L__inference_behavior_model_1_layer_call_and_return_conditional_losses_219419� !"#s�p
Y�V
T�Q
O
8rank2/stimulus_set8�5
inputs/8rank2/stimulus_set���������		
�

trainingp")�&
�
0���������8
� �
1__inference_behavior_model_1_layer_call_fn_218800� !"#l�i
R�O
M�J
H
8rank2/stimulus_set1�.
8rank2/stimulus_set���������		
�

trainingp "����������8�
1__inference_behavior_model_1_layer_call_fn_219072� !"#l�i
R�O
M�J
H
8rank2/stimulus_set1�.
8rank2/stimulus_set���������		
�

trainingp"����������8�
1__inference_behavior_model_1_layer_call_fn_219196� !"#s�p
Y�V
T�Q
O
8rank2/stimulus_set8�5
inputs/8rank2/stimulus_set���������		
�

trainingp "����������8�
1__inference_behavior_model_1_layer_call_fn_219225� !"#s�p
Y�V
T�Q
O
8rank2/stimulus_set8�5
inputs/8rank2/stimulus_set���������		
�

trainingp"����������8�
#__inference_internal_grad_fn_219775������������
���

 
0�-
result_grads_0���������
0�-
result_grads_1���������
0�-
result_grads_2���������
,�)
result_grads_3���������
� "r�o

 
#� 
1���������
#� 
2���������
�
3����������
#__inference_internal_grad_fn_219836������������
���

 
0�-
result_grads_0���������
0�-
result_grads_1���������
0�-
result_grads_2���������
,�)
result_grads_3���������
� "r�o

 
#� 
1���������
#� 
2���������
�
3����������
#__inference_internal_grad_fn_219897������������
���

 
0�-
result_grads_0���������
0�-
result_grads_1���������
0�-
result_grads_2���������
,�)
result_grads_3���������
� "r�o

 
#� 
1���������
#� 
2���������
�
3����������
#__inference_internal_grad_fn_219958������������
���

 
0�-
result_grads_0���������
0�-
result_grads_1���������
0�-
result_grads_2���������
,�)
result_grads_3���������
� "r�o

 
#� 
1���������
#� 
2���������
�
3����������
#__inference_internal_grad_fn_220019������������
���

 
0�-
result_grads_0���������
0�-
result_grads_1���������
0�-
result_grads_2���������
,�)
result_grads_3���������
� "r�o

 
#� 
1���������
#� 
2���������
�
3����������
#__inference_internal_grad_fn_220080������������
���

 
0�-
result_grads_0���������
0�-
result_grads_1���������
0�-
result_grads_2���������
,�)
result_grads_3���������
� "r�o

 
#� 
1���������
#� 
2���������
�
3����������
#__inference_internal_grad_fn_220141������������
���

 
0�-
result_grads_0���������
0�-
result_grads_1���������
0�-
result_grads_2���������
,�)
result_grads_3���������
� "r�o

 
#� 
1���������
#� 
2���������
�
3����������
M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219574� !"#g�d
]�Z
T�Q
O
8rank2/stimulus_set8�5
inputs/8rank2/stimulus_set���������		
p 
� ")�&
�
0���������8
� �
M__inference_rank_similarity_1_layer_call_and_return_conditional_losses_219671� !"#g�d
]�Z
T�Q
O
8rank2/stimulus_set8�5
inputs/8rank2/stimulus_set���������		
p
� ")�&
�
0���������8
� �
2__inference_rank_similarity_1_layer_call_fn_219448� !"#g�d
]�Z
T�Q
O
8rank2/stimulus_set8�5
inputs/8rank2/stimulus_set���������		
p 
� "����������8�
2__inference_rank_similarity_1_layer_call_fn_219477� !"#g�d
]�Z
T�Q
O
8rank2/stimulus_set8�5
inputs/8rank2/stimulus_set���������		
p
� "����������8�
$__inference_signature_wrapper_219167� !"#W�T
� 
M�J
H
8rank2/stimulus_set1�.
8rank2/stimulus_set���������		"7�4
2
output_1&�#
output_1���������8