import 'react-native-gesture-handler';
import { StatusBar } from 'expo-status-bar';
import React, { Component } from 'react';
import { StyleSheet, Text, View, Button, Picker, Image, TextInput, TouchableOpacity} from 'react-native';
import { Octicons, FontAwesome, FontAwesome5, AntDesign, Ionicons, Feather, Entypo } from '@expo/vector-icons';
import { createDrawerNavigator } from '@react-navigation/drawer';
import { NavigationContainer, useNavigationState, useRoute } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import LandingRoutes from './components/landing_routes'
import ImageInput from './components/image_input'
import AnalysisResults from './components/analysis_results'
import AddProducts from './components/add_product'
import EditProduct from './components/edit_product'
import Products from './components/products'
import Product from './components/product'
import Logout from './components/logout';
import Signin from './components/signin';
// secure store import
import * as SecureStore from 'expo-secure-store';

const Drawer = createDrawerNavigator();
const Stack = createStackNavigator();
const focused_color_icon = '#ffffff'
const color_icon = 'silver'
const focused_color_text = 'silver'
const color_text = '#ffffff'

class App extends Component {
    constructor(props) {
        super(props);
        this.state = {
            user_name: '',
            user_access_token: ''
        }

        this.CheckLogin = async () => {
            var should_reload = false

            // get access token
            let token = await SecureStore.getItemAsync('token');
            if (token){
              this.setState({user_access_token: token})
            }else{ should_reload = true }

            // get username
            let user_name = await SecureStore.getItemAsync('user_name');
            if (user_name){
              this.setState({user_name: user_name})
            }else{ should_reload = true }

            if (should_reload == true){  const timeoutId = setTimeout(() => {this.CheckLogin()}, 1000) }
        }

        this.CheckLogout = async () => {
            var should_reload = false

            // get access token
            let token = await SecureStore.getItemAsync('token');
            if (token){
                should_reload = true
            }else{ this.setState({user_access_token: ''}) }

            // get username
            let user_name = await SecureStore.getItemAsync('user_name');
            if (user_name){
                should_reload = true
            }else{ this.setState({user_name: ''}) }

            if (should_reload == true){  const timeoutId = setTimeout(() => {this.CheckLogout()}, 1000) }
        }
    };

    componentDidMount() {
        this.CheckLogin()
        // this.CheckLogout()
    }


    render() {
        var initial_route;
        if (this.state.user_access_token == ''){
            initial_route = 'Landing Routes'
        }else{
            initial_route = 'Image Input'
        }

        return(
          <NavigationContainer>
            <Drawer.Navigator initialRouteName={initial_route} screenOptions={{
                    headerShown: true,
                    drawerStyle: {
                    backgroundColor: '#303030',
                    },
                    drawerActiveTintColor: focused_color_text,
                    drawerInactiveTintColor: color_text
                }}
            >
                <Drawer.Screen name="LandingRoutes" component={LandingRoutes} 
                    options={{
                        headerShown: false,
                        drawerItemStyle: {
                            height: initial_route == 'Landing Routes' ? 50 : 0
                        }
                    }}
                />
                <Drawer.Screen name="Logout" component={Logout}
                    options={{
                        drawerIcon: ({focused}) => (
                            <Entypo name="chevron-right" size={15} color={focused ? focused_color_icon : color_icon}  
                            style={{
                                alignSelf: "center",
                                position: "absolute",
                                right: 5,
                            }}
                            />
                        ),
                        drawerLabel: this.state.user_name,
                        drawerItemStyle: {
                            marginLeft: -20,
                            // marginBottom: 0,
                            height:  initial_route != 'Landing Routes' ? 50 : 0
                        }
                    }}
                />
                <Drawer.Screen name="Image Input" component={ImageInput}
                    options={({ navigation, route }) => ({
                        headerShown: true,
                        drawerIcon: ({focused}) => (
                            <Feather name="image" size={15} color={focused ? focused_color_icon : color_icon}  />
                        ),
                        headerRight: (props) => (
                            <View style={{marginRight: 10}}>
                                <View>
                                    <Feather name="image" size={24} color="#00539C" />
                                </View>
                            </View>
                        ),
                    })}
                />
                <Drawer.Screen name="Analysis Results" component={AnalysisResults} initialParams={{
                    skin_type_match: null,
                    skin_disorder_match: null,
                    recommended_products_list: null,
                    recommended_routine: null,
                    user_allergic_to: null,
                    done: false
                }}
                    options={({ navigation, route }) => ({
                        headerShown: true,
                        drawerIcon: ({focused}) => (
                            <Feather name="list" size={15} color={focused ? focused_color_icon : color_icon}  />
                        ),
                        headerRight: (props) => (
                            <View style={{marginRight: 10}}>
                                <View>
                                    <Feather name="list" size={24} color="#00539C" />
                                </View>
                            </View>
                        ),
                    })}
                />
                <Drawer.Screen name="Add Products" component={AddProducts}
                    options={({ navigation, route }) => ({
                        headerShown: true,
                        drawerIcon: ({focused}) => (
                            <Feather name="shopping-cart" size={15} color={focused ? focused_color_icon : color_icon}  />
                        ),
                        headerRight: (props) => (
                            <View style={{marginRight: 10}}>
                                <View>
                                    <Feather name="shopping-cart" size={24} color="#00539C" />
                                </View>
                            </View>
                        ),
                    })}
                />
                <Drawer.Screen name="Edit Products" component={EditProduct}
                    options={({ navigation, route }) => ({
                        headerShown: true,
                        drawerIcon: ({focused}) => (
                            <Feather name="edit" size={15} color={focused ? focused_color_icon : color_icon}  />
                        ),
                        headerRight: (props) => (
                            <View style={{marginRight: 10}}>
                                <View>
                                    <Feather name="edit" size={24} color="#00539C" />
                                </View>
                            </View>
                        ),
                    })}
                />
                <Drawer.Screen name="Products" component={Products}
                    options={({ navigation, route }) => ({
                        headerShown: true,
                        drawerIcon: ({focused}) => (
                            <Feather name="package" size={15} color={focused ? focused_color_icon : color_icon}  />
                        ),
                        headerRight: (props) => (
                            <View style={{marginRight: 10}}>
                                <View>
                                    <Feather name="package" size={24} color="#00539C" />
                                </View>
                            </View>
                        ),
                    })}
                />
                <Drawer.Screen name="Product" component={Product} initialParams={{product: null}}
                    options={({ navigation, route }) => ({
                        headerShown: true,
                        drawerIcon: ({focused}) => (
                            <Feather name="box" size={15} color={focused ? focused_color_icon : color_icon}  />
                        ),
                        headerRight: (props) => (
                            <View style={{marginRight: 10}}>
                                <View>
                                    <Feather name="box" size={24} color="#00539C" />
                                </View>
                            </View>
                        ),
                        drawerItemStyle: {
                            height: 0
                        }
                    })}
                />
                <Drawer.Screen name="Sign In" component={Signin}
                    options={({ navigation, route }) => ({
                        headerShown: false,
                        drawerIcon: ({focused}) => (
                            <Feather name="image" size={15} color={focused ? focused_color_icon : color_icon}  />
                        ),
                        drawerItemStyle: {
                            height: 0
                        }
                    })}
                />
            </Drawer.Navigator>
          </NavigationContainer>
        );
    }

}

export default App;
