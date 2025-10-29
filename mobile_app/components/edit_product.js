import { StatusBar } from 'expo-status-bar';
import React, { Component }  from 'react';
import { Linking, StyleSheet, Text, View, Button, Picker, Image, TextInput, TouchableOpacity, FlatList, ScrollView} from 'react-native';
import { Octicons, FontAwesome, AntDesign, Ionicons, Feather, Entypo } from '@expo/vector-icons';
import axios from 'axios';
// import {Backend_Url} from './backend_url'
// secure store import
import * as SecureStore from 'expo-secure-store';

class EditProducts extends Component {
    constructor(props) {
        super(props);
        this.state = {
            backend_url: '',
            user_access_token: '',
            product_id: '',
            image: '',
            name: '',
            description: '',
            price: '',
            allergens: '',
            products: []
        }

        this.HandleChange = (value, state) => {
            this.setState({ [state]: value.toString() })
        }

        this.EditProduct = () => {
            var Backend_Url = this.state.backend_url
            var name = this.state.name
            var description = this.state.description
            var price = parseFloat(this.state.price)
            var allergens = this.state.allergens

            if (name == ''){
                alert('Name is required')
            }else if (description == ''){
                alert('Description is required')
            }else if (price == ''){
                alert('Price is required')
            }else{

                var data = new FormData() 
                data.append('user_access_token', this.state.user_access_token)
                data.append('product_id', this.state.product_id)
                data.append('name', name)
                data.append('description', description)
                data.append('price', parseFloat(price))
                data.append('allergens', allergens)
                
                axios.post(Backend_Url + 'editProduct', data)
                .then((res) => {
                    let result = res.data
                    alert('Product updated successfully.')
                    this.setState({product_id: ''})
                    this.GetProducts(Backend_Url)
                }).catch((error) => {
                    alert('Something went wrong, check your connection and try again.')
                })
            }
        }

        this.GetProducts = (backend_url) => {

            var Backend_Url = backend_url

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

        this.SelectProduct = (product) => {
            this.setState({
                product_id: product._id.$oid,
                image: product.image,
                name: product.name,
                description: product.description,
                price: product.price,
                allergens: product.allergens
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
                onStartShouldSetResponder={() => this.SelectProduct(item)}
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

        return<View style={styles.container}>
            <ScrollView style={styles.scroll_view}>
                {
                    this.state.product_id == ''
                    ? <View>
                        {
                            this.state.products.length == 0
                            ? <View>
                                <Text style={{color: '#00539C', fontWeight: 'bold', marginTop: 180, textAlign: 'center'}}>
                                    No products found
                                </Text>
                            </View>
                            : <View style={{marginTop: 20}}>
                                {products_map}
                            </View>
                        }
                    </View>
                    : <View style={{marginBottom: 100}}>                
                        <Image source={{uri: Backend_Url + 'media/' + this.state.image,}} style={{backgroundColor: '#f2f2f2', width: 300, height: 400, marginLeft: 'auto', marginRight: 'auto', marginTop: 20}}/>
                        <TextInput
                            // autoFocus={true}
                            onChangeText={(text) => this.HandleChange(text, "name")}
                            placeholder="Name"
                            placeholderTextColor='grey'
                            value={this.state.name}
                            style={{alignSelf: 'center', width: '100%', borderBottomColor: 'silver', borderBottomWidth: 1, marginTop: 50}}
                        />
                        <TextInput
                            // autoFocus={true}
                            onChangeText={(text) => this.HandleChange(text, "description")}
                            placeholder="Description"
                            placeholderTextColor='grey'
                            value={this.state.description}
                            style={{alignSelf: 'center', width: '100%', borderBottomColor: 'silver', borderBottomWidth: 1, marginTop: 40}}
                        />
                        <TextInput
                            // autoFocus={true}
                            keyboardType = 'number-pad'
                            onChangeText={(text) => this.HandleChange(text, "price")}
                            placeholder="Price"
                            placeholderTextColor='grey'
                            value={this.state.price.toString()}
                            style={{alignSelf: 'center', width: '100%', borderBottomColor: 'silver', borderBottomWidth: 1, marginTop: 40}}
                        />
                        <TextInput
                            // autoFocus={true}
                            onChangeText={(text) => this.HandleChange(text, "allergens")}
                            placeholder="Allergens"
                            placeholderTextColor='grey'
                            value={this.state.allergens}
                            style={{alignSelf: 'center', width: '100%', borderBottomColor: 'silver', borderBottomWidth: 1, marginTop: 40, marginBottom: 100}}
                        />
                        <TouchableOpacity
                            key='back'
                            onPress={() => this.props.navigation.navigate('Products')}
                            style={{backgroundColor: '#ffffff', marginLeft: 'auto', marginRight: 'auto', marginTop: 0, marginBottom: 50, width: '90%', height: 50, borderRadius: 5, borderWidth: 1, borderColor: '#00539C'}}
                        >
                            <Text style={{textAlign: 'center', marginTop: 'auto', marginBottom: 'auto', fontWeight: 'bold', color: '#00539C', fontSize: 17}}>
                                Back
                            </Text>
                        </TouchableOpacity>
                    </View>
                }
            </ScrollView>
            {
                this.state.product_id != ''
                ? <View style={styles.bottom_bar}>
                    <TouchableOpacity
                        key='edit_product'
                        onPress={() => this.EditProduct()}
                        style={{backgroundColor: '#ffffff', marginLeft: 'auto', marginRight: 'auto', marginTop: 0, marginBottom: 0, width: '90%', height: 50, borderRadius: 5, borderWidth: 1, borderColor: '#00539C'}}
                    >
                        <Text style={{textAlign: 'center', marginTop: 'auto', marginBottom: 'auto', fontWeight: 'bold', color: '#00539C', fontSize: 17}}>
                            Edit Product
                        </Text>
                    </TouchableOpacity>
                </View>
                : <View>

                </View>
            }
        </View>
    }

}

export default EditProducts;

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
        flexDirection: 'row',
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