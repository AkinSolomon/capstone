#############################################################################
# Generated by PAGE version 4.7
# in conjunction with Tcl version 8.6
#    Apr 18, 2016 06:40:17 PM


set vTcl(actual_gui_bg) #d9d9d9
set vTcl(actual_gui_fg) #000000
set vTcl(actual_gui_menu_bg) #d9d9d9
set vTcl(actual_gui_menu_fg) #000000
set vTcl(complement_color) #d9d9d9
set vTcl(analog_color_p) #d9d9d9
set vTcl(analog_color_m) #d9d9d9
set vTcl(active_fg) #111111
#############################################################################
# vTcl Code to Load User Fonts

vTcl:font:add_font \
    "-family {DejaVu Sans} -size 20 -weight normal -slant roman -underline 0 -overstrike 0" \
    user \
    vTcl:font10
vTcl:font:add_font \
    "-family {DejaVu Sans} -size 24 -weight normal -slant roman -underline 0 -overstrike 0" \
    user \
    vTcl:font11
#################################
#LIBRARY PROCEDURES
#


if {[info exists vTcl(sourcing)]} {

proc vTcl:project:info {} {
    set base .top36
    namespace eval ::widgets::$base {
        set dflt,origin 0
        set runvisible 1
    }
    namespace eval ::widgets_bindings {
        set tagslist _TopLevel
    }
    namespace eval ::vTcl::modules::main {
        set procs {
        }
        set compounds {
        }
        set projectType single
    }
}
}

#################################
# USER DEFINED PROCEDURES
#

#################################
# GENERATED GUI PROCEDURES
#

proc vTclWindow.top36 {base} {
    if {$base == ""} {
        set base .top36
    }
    if {[winfo exists $base]} {
        wm deiconify $base; return
    }
    set top $base
    ###################
    # CREATING WIDGETS
    ###################
    vTcl::widgets::core::toplevel::createCmd $top -class Toplevel \
        -background {#d9d9d9} -highlightcolor black 
    wm withdraw $top
    wm focusmodel $top passive
    wm geometry $top 1236x754+321+22
    update
    # set in toplevel.wgt.
    global vTcl
    set vTcl(save,dflt,origin) 0
    wm maxsize $top 1905 905
    wm minsize $top 1 1
    wm overrideredirect $top 0
    wm resizable $top 1 1
    wm title $top "New Toplevel 1"
    vTcl:DefineAlias "$top" "Toplevel1" vTcl:Toplevel:WidgetProc "" 1
    ttk::style configure Button -background #d9d9d9
    ttk::style configure Button -foreground #000000
    ttk::style configure Button -font TkDefaultFont
    button $top.but38 \
        -activebackground {#d9d9d9} -activeforeground black \
        -background {#d9d9d9} -font $::vTcl(fonts,vTcl:font11,object) \
        -foreground {#000000} -highlightcolor black -state active -text YES 
    vTcl:DefineAlias "$top.but38" "Button1" vTcl:WidgetProc "Toplevel1" 1
    button $top.but39 \
        -activebackground {#d9d9d9} -activeforeground black \
        -background {#d9d9d9} -font $::vTcl(fonts,vTcl:font11,object) \
        -foreground {#000000} -highlightcolor black -text NO 
    vTcl:DefineAlias "$top.but39" "Button2" vTcl:WidgetProc "Toplevel1" 1
    message $top.mes41 \
        -background {#d9d9d9} -font $::vTcl(fonts,vTcl:font10,object) \
        -foreground {#000000} -highlightcolor black \
        -text {This program will collect your biometric information, including a photo of your face as well as your voice. Using this device conveys your acceptance of these terms. Tap YES to continue.} \
        -width 595 
    vTcl:DefineAlias "$top.mes41" "Message1" vTcl:WidgetProc "Toplevel1" 1
    ###################
    # SETTING GEOMETRY
    ###################
    place $top.but38 \
        -in $top -x 0 -y 420 -width 627 -height 337 -anchor nw \
        -bordermode ignore 
    place $top.but39 \
        -in $top -x 630 -y 420 -width 607 -height 337 -anchor nw \
        -bordermode ignore 
    place $top.mes41 \
        -in $top -x 330 -y 150 -width 595 -height 203 -anchor nw \
        -bordermode ignore 

    vTcl:FireEvent $base <<Ready>>
}

#############################################################################
## Binding tag:  _TopLevel

bind "_TopLevel" <<Create>> {
    if {![info exists _topcount]} {set _topcount 0}; incr _topcount
}
bind "_TopLevel" <<DeleteWindow>> {
    if {[set ::%W::_modal]} {
                vTcl:Toplevel:WidgetProc %W endmodal
            } else {
                destroy %W; if {$_topcount == 0} {exit}
            }
}
bind "_TopLevel" <Destroy> {
    if {[winfo toplevel %W] == "%W"} {incr _topcount -1}
}

Window show .
Window show .top36

