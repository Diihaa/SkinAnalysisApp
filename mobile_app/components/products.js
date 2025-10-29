import { StatusBar } from 'expo-status-bar';
import React, { Component }  from 'react';
import { Platform, StyleSheet, Text, View, Button, Picker, Image, TextInput, TouchableOpacity, FlatList, ScrollView, Alert} from 'react-native';
import { Octicons, FontAwesome, AntDesign, Ionicons, Feather, Entypo, MaterialIcons } from '@expo/vector-icons';
import axios from 'axios';
// import {Backend_Url} from './backend_url'
// SQ-Lite imports
import * as SQLite from 'expo-sqlite'
// import { Backend_Url } from './backend_url';
// secure store import
import * as SecureStore from 'expo-secure-store';

class Products extends Component {
    constructor(props) {
        super(props);
        this.state = {
            backend_url: '',
            products: [],
            user_access_token: '',
            done: false
        }

        this.HandleChange = (value, state) => {
            this.setState({ [state]: value })
        }

        this.GetProducts = (url) => {

            var Backend_Url = url

            var data = new FormData() 
            data.append('user_access_token', null)
            
            axios.post(Backend_Url + 'getAllProducts', data)
            .then((res) => {
                let result = res.data
                this.setState({
                    products: result,
                    done: true
                })
                console.log(result)
            }).catch((error) => {
                console.log(error)
            })
        }

        this.GetBackendURL = async () => {
            var should_reload = false

            // get backend url
            let backend_url = await SecureStore.getItemAsync('backend_url');
            if (backend_url){
              this.setState({backend_url: backend_url})
              this.GetProducts(backend_url) // **************get products ********************
            }else{ should_reload = true }

            if (should_reload == true){  const timeoutId = setTimeout(() => {this.GetBackendURL()}, 1000) }
        }

        this.GetUserDetails = async () => {
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

            if (should_reload == true){  const timeoutId = setTimeout(() => {this.GetUserDetails()}, 1000) }
        }
    };

    componentDidMount() {
        // get backend url
        this.GetBackendURL()
        // get user details
        this.GetUserDetails()

    }

    render() {
        var Backend_Url = this.state.backend_url
        var products_map = this.state.products.map((item, index) => {
            return<View key={'prod'+index}
                onStartShouldSetResponder={() => this.props.navigation.navigate('Product', {product: item})}
                style={{margin: 4, marginBottom: 20, height: 100, flexDirection: 'row', borderBottomColor: '#eaeaea', borderBottomWidth: 1}}
            >
                <View style={{height: 65, flexDirection: 'column', backgroundColor: '#000000', width: 90}}>
                    <Image source={{uri: Backend_Url + 'media/' + item.image,}} style={{width: 90, height: 90}} />
                </View>
                <View style={{flexDirection: 'column', height: 65, borderBottomLeftRadius: 5, borderBottomRightRadius: 5, width: '70%'}}>
                    <View style={{margin: 10}}>
                        {
                            item.name.length <= 23
                            ? <Text style={{textAlign: 'left', fontSize: 15}}>
                                {item.name}
                            </Text>
                            : <Text style={{textAlign: 'left', fontSize: 15}}>
                                {item.name.substring(0, 21)}...
                            </Text>
                        }
                        {
                            item.target_skin_types != ''
                            ? <View>
                                {
                                    item.target_skin_types.length <= 16
                                    ? <Text style={{textAlign: 'left', marginTop: 5, marginLeft: 0, color: '#3f4144', fontSize: 12}}>
                                        Target skin type(s): {item.target_skin_types}
                                    </Text>
                                    : <Text style={{textAlign: 'left', marginTop: 5, marginLeft: 0, color: '#3f4144', fontSize: 12}}>
                                        Target skin type(s): {item.target_skin_types.substring(0, 13)}...
                                    </Text>
                                }
                            </View>
                            : <View>

                            </View>
                        }
                        {
                            item.target_skin_disorders != ''
                            ? <View>
                                {
                                    item.target_skin_disorders.length <= 16
                                    ? <Text style={{textAlign: 'left', marginTop: 5, marginLeft: 0, color: '#3f4144', fontSize: 12}}>
                                        Target skin disorder(s): {item.target_skin_disorders}
                                    </Text>
                                    : <Text style={{textAlign: 'left', marginTop: 5, marginLeft: 0, color: '#3f4144', fontSize: 12}}>
                                        Target skin disorder(s): {item.target_skin_disorders.substring(0, 13)}...
                                    </Text>
                                }
                            </View>
                            : <View>

                            </View>
                        }
                        <Text style={{alignSelf: "flex-end", position: "absolute", right: 5, top: 8}}>
                            ${item.price}
                        </Text>
                    </View>
                </View>
            </View>
        })

        return (
            <View style={styles.container}>
                <ScrollView style={styles.scroll_view}>
                    {
                        this.state.products.length == 0
                        ? <View>
                            <Text style={{color: '#00539C', fontWeight: 'bold', marginTop: 180, textAlign: 'center'}}>
                                No products found
                            </Text>
                        </View>
                        :<View style={{marginTop: 20}}>
                            {products_map}
                        </View>
                    }
                </ScrollView>
            </View>
        );
    }

}

export default Products;

const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: '#ffffff',
    //   textAlign: 'left',
    //   justifyContent: 'center'
    },
    header: {
        height: 40
    },
    search_bar: {
        // borderBottomWidth: 0.1,
        // borderBottomColor: '#eaeaea',
        flexDirection: 'row',
        backgroundColor: '#f2f2f2',
        // alignItems: 'center',
        // justifyContent: 'center',
        height: 50,
        marginTop: 50,
        position: 'absolute', 
        left: 0, 
        right: 0, 
        top: 0
    },
    top_bar: {
        borderBottomWidth: 0.1,
        borderBottomColor: '#eaeaea',
        // flexDirection: 'row',
        backgroundColor: '#ffffff',
        // alignItems: 'center',
        // justifyContent: 'center',
        height: 100,
        position: 'absolute', 
        left: 0, 
        right: 0, 
        top: 0
    },
    scroll_view: {
        height: 2,
        marginTop: 10,
        marginLeft: 10,
        marginRight: 10
    },
    bottom_bar: {
        // flexDirection: 'row',
        backgroundColor: '#ffffff',
        alignItems: 'center',
        justifyContent: 'center',
        height: 70,
        position: 'absolute', 
        left: 0, 
        right: 0, 
        bottom: 0
    }
});