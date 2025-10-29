import { StatusBar } from 'expo-status-bar';
import React, { Component }  from 'react';
import { StyleSheet, Text, View, Button, Picker, Image, TextInput, TouchableOpacity, FlatList, ScrollView} from 'react-native';
import { Octicons, FontAwesome, AntDesign, Ionicons, Feather, Entypo } from '@expo/vector-icons';
import axios from 'axios';
import {Backend_Url} from './backend_url'
import { createStackNavigator } from '@react-navigation/stack';
import Landing from './landing'
import Signin from './signin'
import Signup from './signup'
import UrlSetup from './url_setup'

const Stack = createStackNavigator();

class LandingRoutes extends Component {
    constructor(props) {
        super(props);
        this.state = {
            
        }

        // this.CheckLogin_ = () => {
        //     const { CheckLogin } = this.props.route.params;
        //     CheckLogin()
        // }
    };

    componentDidMount() {

    }


    render() {

        return<Stack.Navigator initialRouteName='URL Setup'
            screenOptions={({ navigation, route }) =>({
                headerShown: true
            })}
        >
            <Stack.Screen name="URL Setup" component={UrlSetup}
                options={({ navigation, route }) =>({
                    headerShown: false
                })}
            />
            <Stack.Screen name="Landing" component={Landing}
                options={({ navigation, route }) =>({
                    headerShown: false
                })}
            />
            <Stack.Screen name="Signin" component={Signin}
                options={({ navigation, route }) =>({
                    headerShown: true
                })}
            />
            <Stack.Screen name="Signup" component={Signup}
                options={({ navigation, route }) =>({
                    headerShown: true
                })}
            />
        </Stack.Navigator>
    }

}

export default LandingRoutes;

const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: '#ffffff',
    //   textAlign: 'left',
    //   justifyContent: 'center'
    },
    scroll_view: {
        height: 2
    },
    bottom_bar: {
        borderTopWidth: 1,
        borderTopColor: 'gold',
        flexDirection: 'row',
        backgroundColor: '#ffffff',
        alignItems: 'center',
        justifyContent: 'center',
        height: 60,
        position: 'absolute', 
        left: 0, 
        right: 0, 
        bottom: 0
    }
});