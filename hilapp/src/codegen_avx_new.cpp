//------------------------------------------------------------------------------
// Generate transformed 
// hardware-dependent "kernels".
//
// Uses Clang RecursiveASTVisitor and Rewriter 
// interfaces
//
// Kari Rummukainen 2017-18
// 
//------------------------------------------------------------------------------
#include <sstream>
#include <string>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"
//#include "llvm/Support/raw_ostream.h"

#include "hilapp.h"
#include "myastvisitor.h"
#include "stringops.h"

extern std::string looping_var;
extern std::string parity_name;

extern std::string parity_in_this_loop;






/// An AST walker for finding and handling variable declarations
/// in a loop function
class LoopFunctionHandler : public GeneralVisitor, public RecursiveASTVisitor<LoopFunctionHandler> {
public:
  using GeneralVisitor::GeneralVisitor;

  //Buffer for the function copy
  srcBuf functionBuffer;
  int vector_size;

  bool TraverseStmt(Stmt *s){
    RecursiveASTVisitor<LoopFunctionHandler>::TraverseStmt(s);
    return true;
  }
  bool VisitVarDecl(VarDecl *var);
  bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr *op);
  bool VisitBinaryOperator(BinaryOperator *op);
};


bool LoopFunctionHandler::VisitVarDecl(VarDecl *var){
  std::string typestring = var->getType().getAsString();
  
  size_t begin;
  begin = typestring.rfind("element",0);
  if(begin != std::string::npos){
    // This variable is an element, replace with vector
    std::string vector_type = typestring;
    if(begin != std::string::npos){
      vector_type.replace(begin, 7, "vectorize_struct");
      vector_type.replace(vector_type.find_last_of(">"), 1, ", "+std::to_string(vector_size)+">::type");
    }

    if( var->isDirectInit() ){
      std::string init = TheRewriter.getRewrittenText(var->getInit()->getSourceRange());
      functionBuffer.replace(var->getSourceRange(), vector_type+" "+var->getNameAsString() + "=" + init );
    } else {
      functionBuffer.replace(var->getSourceRange(), vector_type+" "+var->getNameAsString());
    }

  } else {
    if(var->hasInit()){
      LoopAssignChecker lac(TheRewriter, Context);
      lac.TraverseStmt(var->getInit());
    }
  }
  return true;
}


bool LoopFunctionHandler::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *op){
  if(op && op->isAssignmentOp()){
    std::string type = op->getArg(0)->getType().getAsString();
    type = remove_all_whitespace(type);
    if(type.rfind("element<",0) == std::string::npos){
      LoopAssignChecker lac(TheRewriter, Context);
      lac.TraverseStmt(op);
    }
  }
  return true;
}

bool LoopFunctionHandler::VisitBinaryOperator(BinaryOperator *op){
  if(op && op->isAssignmentOp()){
    std::string type = op->getLHS()->getType().getAsString();
    type = remove_all_whitespace(type);
    if(type.rfind("element<",0) == std::string::npos){
      LoopAssignChecker lac(TheRewriter, Context);
      lac.TraverseStmt(op);
    }
  }
  return true;
}




/// Replace element types with vector. Leaves other types untouched.
static void replace_element_with_vector(SourceRange sr, std::string typestring, std::string namestring, int vector_size, srcBuf &functionBuffer){
  if(typestring.rfind("element",0) != std::string::npos){
    std::string vector_type = typestring;
    size_t begin;
    begin = vector_type.find("element");
    if(begin != std::string::npos){
      vector_type.replace(begin, 7, "vectorize_struct");
      vector_type.replace(vector_type.find_last_of(">"), 1, ", "+std::to_string(vector_size)+">::type");
    }
    
    functionBuffer.replace(sr, vector_type+" "+namestring);
  }
}


/// This should allow calling with element<>-type parameters from
/// loops. Generate a copy with elements replaced with vectors.
void MyASTVisitor::handle_loop_function_avx(FunctionDecl *fd) {
  SourceRange sr = fd->getSourceRange();
  srcBuf * sourceBuf = get_file_srcBuf( sr.getBegin() );
  PrintingPolicy pp(Context->getLangOpts());


  // Track wether the function actually contains elements.
  // if not, no new function should be written
  bool generate_function = false;

  // Check allowed vector sizes
  int smallest=1, largest=0;
  for( clang::ParmVarDecl *par : fd->parameters() ){
    std::string typestring = par->getType().getAsString(pp);
    if(typestring.find("double") != std::string::npos){
      smallest = 4;
      largest = 8;
    }
    if( typestring.find("float") != std::string::npos ||
        typestring.find("int") != std::string::npos || 
        typestring.find("coordinate_vector") != std::string::npos ){
      smallest = 8;
      largest = 16;
    }

    // Check if there are elements in the first place
    if(typestring.find("element") != std::string::npos)
      generate_function = true;

  }

  if(generate_function) for( int vector_size = smallest; vector_size <= largest; vector_size*=2 ){
    LoopFunctionHandler lfh(TheRewriter, Context);
    lfh.functionBuffer.copy_from_range(sourceBuf,sr);
    lfh.vector_size = vector_size;

    // Handle each parameter
    for( clang::ParmVarDecl *par : fd->parameters() ){
      std::string typestring = par->getType().getAsString(pp);
      replace_element_with_vector(par->getSourceRange(), typestring, par->getNameAsString(), vector_size, lfh.functionBuffer);
    }

    // Handle return type
    // Note: C++ cannot specialize only based on return type. Therefore we
    // only write a new function if the parameters contain elements
    std::string typestring = fd->getReturnType().getAsString(pp);
    replace_element_with_vector(fd->getReturnTypeSourceRange(), typestring, "", vector_size, lfh.functionBuffer);

    lfh.TraverseStmt(fd->getBody());

    std::string buffer = lfh.functionBuffer.dump();
    if( !(fd->hasBody()) ){
      // Declaration does not contain a body, needs a semicolon
      buffer += ";";
    }
    buffer += "\n";
    sourceBuf->insert(sr.getBegin(), buffer, true, true);
  }
}

///////////////////////////////////////////////////////////////////////////////////
/// Check that
///  a) no site dependent conditional
///  b) fields have the same vector size (number size)
///  c) variables are vectorizable to the same size
///////////////////////////////////////////////////////////////////////////////////

bool MyASTVisitor::check_loop_vectorizable(Stmt *S, int & vector_size, std::string & diag_str) {

  vector_size = 0;
  bool is_vectorizable = true;
  
  int diag_count = 0;
  std::string reason;
  
  // check if loop has conditional
  is_vectorizable = !loop_info.has_site_dependent_conditional;
  if (!is_vectorizable) {
    reason = "it contains site dependent conditional";
    diag_count++;
  }

  // check if the fields are vectorizable in a compatible manner
  if (field_info_list.size() > 0) {
    for (field_info & fi : field_info_list) {
      if (!fi.vecinfo.is_vectorizable) {
        is_vectorizable = false;
        if (diag_count++ > 0) reason += "\n";
        reason += "field variable '" + fi.old_name + "' is not vectorizable";
      } else {
        if (vector_size == 0) vector_size = fi.vecinfo.vector_size;
        else if (fi.vecinfo.vector_size != vector_size) {
          is_vectorizable = false;

          if (diag_count++ > 0) reason += "\n";
          reason += "vector size of variables '" + fi.old_name + "' is " 
              + std::to_string(fi.vecinfo.vector_size) + " and '" + field_info_list.begin()->old_name 
              + "' is " + std::to_string(vector_size);
        }
      }
    }
  }

  // and then if the site dep. variables are vectorizable
  if (var_info_list.size() > 0) {
    for (var_info & vi : var_info_list) if (vi.is_site_dependent) {
      if (vi.vecinfo.is_vectorizable) {
        if (vector_size == 0) vector_size = vi.vecinfo.vector_size;
        else if (vector_size != vi.vecinfo.vector_size) {
          is_vectorizable = false;

          if (diag_count++ > 0) reason += "\n";
          reason += "vector size of variables '" + vi.name + "' is " 
            + std::to_string(vi.vecinfo.vector_size) + " and '" + field_info_list.begin()->old_name 
            + "' is " + std::to_string(vector_size);
        }
      } else {
        is_vectorizable = false;

        if (diag_count++ > 0) reason += "\n";
        reason += "variable '" + vi.name + "' is not vectorizable";
      }
    }
  }

  if (!is_vectorizable) {
    diag_str = "Loop is not AVX vectorizable because " + reason;
    if( cmdline::avx_info > 0 || cmdline::verbosity > 0) 
      reportDiag(DiagnosticsEngine::Level::Remark, S->getSourceRange().getBegin(), 
                 "Loop is not AVX vectorizable because %0", reason.c_str());

  } else {
    diag_str = "Loop is AVX vectorizable";

    if (cmdline::avx_info > 1 || cmdline::verbosity > 1) 
      reportDiag(DiagnosticsEngine::Level::Remark, S->getSourceRange().getBegin(), 
                 "Loop is AVX vectorizable");
  }

  return is_vectorizable;
}

///////////////////////////////////////////////////////////////////////////////////////////
///  Main entry for AVX loop generation
///////////////////////////////////////////////////////////////////////////////////////////

std::string MyASTVisitor::generate_code_avx(Stmt *S, bool semicolon_at_end, srcBuf & loopBuf, bool generate_wait_loops) {

  std::stringstream code;
  int vector_size;

  // is the loop vectorizable?
  std::string vector_diag;
  bool is_vectorized = check_loop_vectorizable(S, vector_size, vector_diag);

  code << comment_string(vector_diag) << '\n';

  // can go through std non-vector code generation 
  if (!is_vectorized) {
    code << generate_code_cpu(S, semicolon_at_end, loopBuf, generate_wait_loops);
    return code.str();
  }

  // Create temporary variables for reductions (vector reduction is in the loop)
  for (var_info & v : var_info_list) {
    if (v.reduction_type != reduction::NONE) {
      v.new_name = "v_" + v.reduction_name;
      // Allocate memory for a reduction. This will be filled in the kernel
      code << v.vecinfo.vectorized_type << ' ' << v.new_name;
      if (v.reduction_type == reduction::SUM) 
        code << "(0);\n";
      else if (v.reduction_type == reduction::PRODUCT)
        code << "(1);\n";
    }
  }

  // Set loop lattice for neibhbour arrays
  std::string fieldname = field_info_list.front().new_name;
  code << "const auto * RESTRICT loop_lattice = "
       << fieldname << ".fs->vector_lattice;\n";

  // Set the start and end points
  code << "const int loop_begin = loop_lattice->loop_begin(" << parity_in_this_loop << ");\n";
  code << "const int loop_end   = loop_lattice->loop_end(" << parity_in_this_loop << ");\n";

  if (generate_wait_loops) {
    code << "for (int _wait_i_ = 0; _wait_i_ < 2; ++_wait_i_) {\n";
  }

  // Start the loop
  code << "for(int " << looping_var <<" = loop_begin; " 
       << looping_var << " < loop_end; ++" << looping_var << ") {\n";

  if (generate_wait_loops) {
    code << "if (((loop_lattice->vec_wait_arr_[" << looping_var << "] & _dir_mask_) != 0) == _wait_i_) {\n";
  }

  // Add vector reduction variable here, inside the loop
  for (vector_reduction_ref & vrf : vector_reduction_ref_list) {
    // Allocate memory for a reduction and initialize
    if (vrf.reduction_type == reduction::SUM) {
      code << "vectorize_struct<" <<vrf.type << ", vector_size>::type v_" 
           << vrf.vector_name << "(0);\n";
    }
    if (vrf.reduction_type == reduction::PRODUCT) {
      code << "vectorize_struct<" <<vrf.type << ", vector_size>::type v_" 
           << vrf.vector_name << "(1);\n";
    }
    loopBuf.replace( vrf.ref, "v_"+vrf.vector_name );
  }

  
  // Create temporary field element variables
  for (field_info & l : field_info_list) {

    // First check for direction references. If any found, create list of temp
    // variables
    if (l.is_read_nb) {
      for (dir_ptr & d : l.dir_list) if(d.count > 0){
        std::string dirname;
        if (d.is_constant_direction) dirname = d.direxpr_s;  // orig. string
        else dirname = remove_X( loopBuf.get(d.parityExpr->getSourceRange()) ); // mapped name was get_stmt_str(d.e);

        code << l.vecinfo.vectorized_type << " " 
             << d.name_with_dir
             << " = " << l.new_name << ".get_vector_at<" << l.vecinfo.vectorized_type << ">(loop_lattice->neighbours[" 
             << dirname << "][" << looping_var << "]);\n";

        // and replace references in loop body
        for (field_ref * ref : d.ref_list) {
          loopBuf.replace(ref->fullExpr, d.name_with_dir);
        }
      } 
    }

    if (l.is_read_atX) {
      code << l.vecinfo.vectorized_type << " "
           << l.loop_ref_name << " = " 
           << l.new_name << ".get_vector_at<" << l.vecinfo.vectorized_type << ">(" << looping_var << ");\n";
    } else if (l.is_written) {
      code << l.vecinfo.vectorized_type << " "
           << l.loop_ref_name << ";\n";
    }

    // and finally replace these references in body 
    for (field_ref * ref : l.ref_list) if (!ref->is_direction) {
      loopBuf.replace(ref->fullExpr, l.loop_ref_name);
    }
  }

  // other variable refs
  for ( var_info & vi : var_info_list ) {
    // reduction variables
    if (vi.reduction_type != reduction::NONE) {
      // Replace references in the loop body
      for (var_ref & vr : vi.refs) {
        loopBuf.replace( vr.ref, vi.new_name );
      }
    } else if (vi.is_site_dependent) {
      // now must be loop-local vectorized var
      // change declaration - name need not be changed
      // loopBuf.replace( vi.decl->getSourceRange(), vi.vecinfo.vectorized_type );
      loopBuf.replace( vi.decl->getTypeSourceInfo()->getTypeLoc().getSourceRange(),
                       vi.vecinfo.vectorized_type );
    }
  }

  // Handle calls to special in-loop functions
  for ( special_function_call & sfc : special_function_call_list ){
    std::string repl = sfc.replace_expression+"(";
    if ( sfc.args != "" ) {
      repl += sfc.args;
      if ( sfc.add_loop_var ) repl += ", ";
    }
    if ( sfc.add_loop_var ) repl += looping_var;
    repl += ")";

    loopBuf.replace(sfc.fullExpr, repl);
  }

  // Vector reductions must be in the sames scope as the loop body. Otherwise the index may be undefined. 
  // Therefore add it before the closing }
  if (!semicolon_at_end){
    // Remove the last }
    loopBuf.remove(loopBuf.size()-2, loopBuf.size()-1);
  }

  // Dump the main loop code here
  code << loopBuf.dump();
  if (semicolon_at_end) code << ";";
  code << "\n";


  // Add vector reductions
  int i=0;
  for (vector_reduction_ref & vrf : vector_reduction_ref_list) {
    // run reduction over the vector
    code << "int v_index_" << i << "[vector_size];\n";
    code << vrf.index_name << ".store(&v_index_" << i << "[0]);\n";
    code << vrf.type << " a_" << vrf.vector_name << "[vector_size];\n";
    code << "v_" << vrf.vector_name << ".store(&a_" << vrf.vector_name << "[0]);\n";
    code << "for( int i=0; i<vector_size; i++){\n";
    if (vrf.reduction_type == reduction::SUM) {
      code << vrf.vector_name << "[v_index_" << i << "[i]] += " 
           << "a_" << vrf.vector_name << "[i];\n";
    }
    if (vrf.reduction_type == reduction::PRODUCT) {
      code << vrf.vector_name << "[v_index_" << i << "[i]] *= " 
           << "a_" << vrf.vector_name << "[i];\n";
    }
    code << "}\n";
  }

  if (!semicolon_at_end){
    code << "}";
  }
  code << "\n";


  // Add calls to setters 
  for (field_info & l : field_info_list){
    if(l.is_written) {
      code << l.new_name << ".set_vector_at<" << l.vecinfo.vectorized_type << ">(" 
           << l.loop_ref_name << ", " << looping_var << ");\n";
    }
  }

  code << "}\n";


  if (generate_wait_loops) {
    // add the code for 2nd round
    code << "}\nif (_dir_mask_ == 0) break;    // No need for another round\n";
    
    for (field_info & l : field_info_list) {
      // If neighbour references exist, communicate them
      for (dir_ptr & d : l.dir_list) if(d.count > 0){
        code << l.new_name << ".wait_get("
             << d.direxpr_s << ", " << parity_in_this_loop << ");\n";
      }
    }
    code << "}\n";

  }


  // Final reduction of the temporary reduction variables
  for (var_info & v : var_info_list) {
    if (v.reduction_type == reduction::SUM) {
      code << v.reduction_name << " = reduce_sum(" << v.new_name << ");\n";
    } else if (v.reduction_type == reduction::PRODUCT) {
      code << v.reduction_name << " = reduce_prod(" << v.new_name << ");\n";
    }
  }

  return code.str();
}



